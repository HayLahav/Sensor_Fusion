"""Check a planned ego path against the perception BEV.

A planner (e.g. Qwen-Drive) proposes a path from camera images. This module
checks that path against what the LiDAR + tracking stack sees, on the same
BEV grid layout as occupancy/bev_grid.py (rows = forward z, cols = lateral x,
camera frame).

Three checks, each reported per waypoint:
  obstacle  — the ego footprint overlaps a static obstacle: LiDAR returns
              above the ground (ground removed by Utonia labels, or by height
              above the local lowest return), plus the boxes of tracks that
              are not moving (parked cars).
  off_road  — the ego centre is on a cell whose semantic class is known and
              not drivable (camera semantics from the BEV grid, or Utonia
              LiDAR classes).
  agent     — the ego footprint comes too close to a moving track's predicted
              position at the same time (constant-velocity forecast). Only
              tracks the motion gate marked "moving" are forecast.

Why not use OccupancyGrid.grid directly: it marks every LiDAR return as
occupied, including the road surface, so almost any path would collide.

Ego footprint: three discs along the heading (front, centre, rear), a common
approximation for a car. The ego origin is the camera, which on KITTI-360 is
near the middle of the roof.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math
import cv2
import numpy as np

from fusion_perception.segmentation.lidar_classes import CLASS_NAMES as LIDAR_CLASSES
from fusion_perception.utils.dataclasses import OccupancyGrid, Track

# Cityscapes trainIds in OccupancyGrid.sem_grid
_CS_DRIVABLE = frozenset({0})                          # road
_CS_NON_DRIVABLE = frozenset({1, 2, 3, 4, 8, 9})       # sidewalk, building, wall, fence, vegetation, terrain
# Utonia LiDAR classes (segmentation/lidar_classes.py)
_L = {n: i for i, n in enumerate(LIDAR_CLASSES)}
_LIDAR_GROUND = frozenset({_L["road"], _L["sidewalk"], _L["other-ground"], _L["terrain"]})
_LIDAR_DRIVABLE = frozenset({_L["road"], _L["other-ground"]})   # other-ground includes parking
_LIDAR_NON_DRIVABLE = frozenset({_L["sidewalk"], _L["terrain"], _L["building"],
                                 _L["fence-wall"], _L["vegetation"]})

DRIVABLE, NON_DRIVABLE, UNKNOWN_SURFACE = 1, 0, -1


@dataclass
class BEVCheckConfig:
    ego_disc_radius: float = 1.0          # m, radius of each footprint disc
    ego_disc_offsets: tuple = (-1.3, 0.0, 1.3)   # m along heading (rear, centre, front)
    agent_radius: float = 1.0             # m, assumed radius of a moving agent
    agent_margin: float = 0.5             # m, extra clearance required from agents
    obstacle_min_height: float = 0.3      # m above local ground to count as obstacle
    obstacle_max_height: float = 2.5      # m above local ground (ignore overhanging trees/signs)
    ground_window: float = 2.0            # m, window for local ground height estimate
    min_obstacle_points: int = 2          # LiDAR points per cell to mark an obstacle
    off_road_neighbourhood: int = 1       # cells; majority vote radius for surface class
    ignore_first_s: float = 0.0           # skip violations before this time (s)


@dataclass
class AgentForecast:
    track_id: int
    class_name: str
    x: float                              # camera frame, metres
    z: float
    vx: float                             # m/s, ego-compensated
    vz: float


@dataclass
class BEVPlanContext:
    resolution: float
    x_range: list[float]
    z_range: list[float]
    obstacle: np.ndarray                  # bool [rows, cols]
    surface: np.ndarray                   # int8 [rows, cols]: 1 drivable, 0 not, -1 unknown
    agents: list[AgentForecast] = field(default_factory=list)

    def cell(self, x: float, z: float) -> Optional[tuple[int, int]]:
        if not (self.x_range[0] <= x < self.x_range[1] and self.z_range[0] <= z < self.z_range[1]):
            return None
        return int((z - self.z_range[0]) / self.resolution), int((x - self.x_range[0]) / self.resolution)


@dataclass
class Violation:
    kind: str                             # obstacle | off_road | agent
    waypoint: int
    t: float
    detail: str


@dataclass
class BEVCheckResult:
    ok: bool
    violations: list[Violation]
    first_violation_t: Optional[float]
    min_agent_clearance: Optional[float]  # m (negative = overlap), None if no moving agents
    waypoints_checked: int
    waypoints_outside_grid: int

    def summary(self) -> str:
        if self.ok:
            return "ok"
        kinds = sorted({v.kind for v in self.violations})
        return f"{', '.join(kinds)} from t={self.first_violation_t:.1f}s"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "summary": self.summary(),
            "first_violation_t": self.first_violation_t,
            "min_agent_clearance": self.min_agent_clearance,
            "waypoints_checked": self.waypoints_checked,
            "waypoints_outside_grid": self.waypoints_outside_grid,
            "violations": [v.__dict__ for v in self.violations],
        }


# ── Building the context ─────────────────────────────────────────────────────

def _grid_shape(x_range, z_range, res) -> tuple[int, int]:
    return (int(round((z_range[1] - z_range[0]) / res)),
            int(round((x_range[1] - x_range[0]) / res)))


def _obstacles_from_lidar(
    pts_cam: np.ndarray,
    labels: Optional[np.ndarray],
    shape: tuple[int, int],
    x_range, z_range, res: float,
    cfg: BEVCheckConfig,
) -> np.ndarray:
    rows_n, cols_n = shape
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    keep = (x >= x_range[0]) & (x < x_range[1]) & (z >= z_range[0]) & (z < z_range[1])
    if labels is not None:
        keep &= ~np.isin(labels, list(_LIDAR_GROUND)) & (labels != 255)
    x, y, z = x[keep], y[keep], z[keep]
    if len(x) == 0:
        return np.zeros(shape, dtype=bool)
    r = ((z - z_range[0]) / res).astype(np.intp).clip(0, rows_n - 1)
    c = ((x - x_range[0]) / res).astype(np.intp).clip(0, cols_n - 1)

    height = -y                                         # camera y points down
    if labels is None:
        # Local ground = lowest return in a window around each cell. Built
        # from all in-range points so ground under obstacles is found.
        low = np.full(shape, np.inf, dtype=np.float32)
        np.minimum.at(low, (r, c), height.astype(np.float32))
        k = max(1, int(round(cfg.ground_window / res)))
        low_f = np.where(np.isfinite(low), low, 1e6).astype(np.float32)
        ground = -cv2.dilate(-low_f, np.ones((2 * k + 1, 2 * k + 1), np.uint8))   # windowed min
        rel = height - ground[r, c]
        above = (rel > cfg.obstacle_min_height) & (rel < cfg.obstacle_max_height)
        r, c = r[above], c[above]

    counts = np.zeros(shape, dtype=np.int32)
    np.add.at(counts, (r, c), 1)
    return counts >= cfg.min_obstacle_points


def _footprint_poly(box_3d, x_range, z_range, res) -> Optional[np.ndarray]:
    """Box [cx,cy,cz,w,h,l,ry] → [4,2] int32 (col,row) polygon, same rotation
    convention as visualization/bev_renderer.py."""
    cx, _, cz, w, _, l, ry = box_3d
    hw, hl = max(w, 0.3) / 2, max(l, 0.3) / 2
    local = np.array([[-hw, -hl], [hw, -hl], [hw, hl], [-hw, hl]])
    cr, sr = math.cos(ry), math.sin(ry)
    world = local @ np.array([[cr, -sr], [sr, cr]]).T + [cx, cz]
    cols = (world[:, 0] - x_range[0]) / res
    rows = (world[:, 1] - z_range[0]) / res
    return np.round(np.stack([cols, rows], 1)).astype(np.int32)


def _surface_layer(
    occ: Optional[OccupancyGrid],
    pts_cam: Optional[np.ndarray],
    labels: Optional[np.ndarray],
    shape, x_range, z_range, res,
) -> np.ndarray:
    surface = np.full(shape, UNKNOWN_SURFACE, dtype=np.int8)
    if occ is not None and occ.sem_grid is not None:
        sem = np.asarray(occ.sem_grid, dtype=np.int16)
        if sem.shape == shape:
            surface[np.isin(sem, list(_CS_DRIVABLE))] = DRIVABLE
            surface[np.isin(sem, list(_CS_NON_DRIVABLE))] = NON_DRIVABLE
    if pts_cam is not None and labels is not None and len(pts_cam):
        # LiDAR labels cover all around the car; camera ones only its view.
        x, z = pts_cam[:, 0], pts_cam[:, 2]
        keep = (x >= x_range[0]) & (x < x_range[1]) & (z >= z_range[0]) & (z < z_range[1])
        r = ((z[keep] - z_range[0]) / res).astype(np.intp).clip(0, shape[0] - 1)
        c = ((x[keep] - x_range[0]) / res).astype(np.intp).clip(0, shape[1] - 1)
        lab = labels[keep]
        drv = np.isin(lab, list(_LIDAR_DRIVABLE))
        non = np.isin(lab, list(_LIDAR_NON_DRIVABLE))
        surface[r[non], c[non]] = NON_DRIVABLE
        surface[r[drv], c[drv]] = DRIVABLE      # drivable wins on mixed cells
    return surface


def build_plan_context(
    tracks: list[Track],
    occupancy: Optional[OccupancyGrid] = None,
    lidar_pts_cam: Optional[np.ndarray] = None,
    lidar_labels: Optional[np.ndarray] = None,
    resolution: Optional[float] = None,
    x_range: Optional[list[float]] = None,
    z_range: Optional[list[float]] = None,
    cfg: Optional[BEVCheckConfig] = None,
) -> BEVPlanContext:
    """Build obstacle / surface / agent layers for one frame.

    Grid geometry comes from `occupancy` when given, else from the arguments.
    lidar_pts_cam: [N,3] rectified camera frame; lidar_labels: optional [N]
    Utonia classes aligned with it.
    """
    cfg = cfg or BEVCheckConfig()
    if occupancy is not None:
        resolution, x_range, z_range = occupancy.resolution, list(occupancy.x_range), list(occupancy.z_range)
    if resolution is None or x_range is None or z_range is None:
        raise ValueError("Give an OccupancyGrid or resolution/x_range/z_range")
    shape = _grid_shape(x_range, z_range, resolution)

    obstacle = np.zeros(shape, dtype=bool)
    if lidar_pts_cam is not None and len(lidar_pts_cam):
        obstacle |= _obstacles_from_lidar(lidar_pts_cam, lidar_labels, shape,
                                          x_range, z_range, resolution, cfg)

    agents: list[AgentForecast] = []
    moving_mask = np.zeros(shape, dtype=np.uint8)
    for t in tracks:
        if not t.is_active or t.last_box_3d is None:
            continue
        poly = _footprint_poly(t.last_box_3d, x_range, z_range, resolution)
        if getattr(t, "motion_state", "unknown") == "moving":
            cx, _, cz = t.last_box_3d[:3]
            agents.append(AgentForecast(t.track_id, t.class_name, float(cx), float(cz),
                                        float(t.velocity_3d[0]), float(t.velocity_3d[2])))
            cv2.fillPoly(moving_mask, [poly], 1)
        else:
            # Parked / not yet known to move → static obstacle
            tmp = np.zeros(shape, dtype=np.uint8)
            cv2.fillPoly(tmp, [poly], 1)
            obstacle |= tmp.astype(bool)
    # A moving agent's current LiDAR returns are handled by the agent check
    obstacle &= ~moving_mask.astype(bool)

    surface = _surface_layer(occupancy, lidar_pts_cam, lidar_labels, shape,
                             x_range, z_range, resolution)
    return BEVPlanContext(resolution, x_range, z_range, obstacle, surface, agents)


# ── Checking a trajectory ────────────────────────────────────────────────────

def ego_to_cam_xz(traj: np.ndarray) -> np.ndarray:
    """[N,≥2] ego (forward, left) → [N,2] camera (x, z)."""
    return np.stack([-traj[:, 1], traj[:, 0]], axis=1)


def _disc_centres(traj: np.ndarray, offsets) -> np.ndarray:
    """[N,3] ego (x, y, heading) → [N, D, 2] disc centres in camera (x, z)."""
    h = traj[:, 2]
    fwd = np.stack([np.cos(h), np.sin(h)], 1)              # ego frame unit vector
    centres = traj[:, None, :2] + np.asarray(offsets)[None, :, None] * fwd[:, None, :]
    flat = centres.reshape(-1, 2)
    return ego_to_cam_xz(flat).reshape(len(traj), len(offsets), 2)


def _disc_hits(mask: np.ndarray, ctx: BEVPlanContext, cx: float, cz: float, rad: float) -> bool:
    k = int(math.ceil(rad / ctx.resolution))
    cell = ctx.cell(cx, cz)
    if cell is None:
        return False
    r0, c0 = cell
    rs = slice(max(r0 - k, 0), min(r0 + k + 1, mask.shape[0]))
    cs = slice(max(c0 - k, 0), min(c0 + k + 1, mask.shape[1]))
    sub = mask[rs, cs]
    if not sub.any():
        return False
    rr, cc = np.nonzero(sub)
    # cell centres in metres
    xs = ctx.x_range[0] + (cc + cs.start + 0.5) * ctx.resolution
    zs = ctx.z_range[0] + (rr + rs.start + 0.5) * ctx.resolution
    return bool((np.hypot(xs - cx, zs - cz) <= rad + ctx.resolution / 2).any())


def _surface_at(ctx: BEVPlanContext, x: float, z: float, n: int) -> int:
    cell = ctx.cell(x, z)
    if cell is None:
        return UNKNOWN_SURFACE
    r0, c0 = cell
    sub = ctx.surface[max(r0 - n, 0):r0 + n + 1, max(c0 - n, 0):c0 + n + 1]
    known = sub[sub != UNKNOWN_SURFACE]
    if known.size == 0:
        return UNKNOWN_SURFACE
    return DRIVABLE if (known == DRIVABLE).sum() >= (known == NON_DRIVABLE).sum() else NON_DRIVABLE


def check_trajectory(
    traj: np.ndarray,
    ctx: BEVPlanContext,
    dt: float = 0.1,
    cfg: Optional[BEVCheckConfig] = None,
) -> BEVCheckResult:
    """Check [N,3] ego-frame waypoints (x fwd, y left, heading) at t = dt·(i+1)."""
    cfg = cfg or BEVCheckConfig()
    traj = np.asarray(traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] < 3:
        raise ValueError("traj must be [N,3] (x, y, heading)")
    discs = _disc_centres(traj, cfg.ego_disc_offsets)
    centre = ego_to_cam_xz(traj)

    violations: list[Violation] = []
    outside = 0
    min_clear: Optional[float] = None
    for i in range(len(traj)):
        t = dt * (i + 1)
        if ctx.cell(*centre[i]) is None:
            outside += 1
            continue
        if t < cfg.ignore_first_s:
            continue
        for d in discs[i]:
            if _disc_hits(ctx.obstacle, ctx, d[0], d[1], cfg.ego_disc_radius):
                violations.append(Violation("obstacle", i, t, f"near x={d[0]:.1f} z={d[1]:.1f}"))
                break
        if _surface_at(ctx, centre[i, 0], centre[i, 1], cfg.off_road_neighbourhood) == NON_DRIVABLE:
            violations.append(Violation("off_road", i, t,
                                        f"x={centre[i, 0]:.1f} z={centre[i, 1]:.1f}"))
        for a in ctx.agents:
            ax, az = a.x + a.vx * t, a.z + a.vz * t
            gap = float(np.min(np.hypot(discs[i, :, 0] - ax, discs[i, :, 1] - az))) \
                - cfg.ego_disc_radius - cfg.agent_radius
            min_clear = gap if min_clear is None else min(min_clear, gap)
            if gap < cfg.agent_margin:
                violations.append(Violation("agent", i, t,
                                            f"track {a.track_id} ({a.class_name}) gap {gap:.1f} m"))

    first = min((v.t for v in violations), default=None)
    return BEVCheckResult(
        ok=not violations,
        violations=violations,
        first_violation_t=first,
        min_agent_clearance=min_clear,
        waypoints_checked=len(traj) - outside,
        waypoints_outside_grid=outside,
    )


def draw_trajectory(
    bev_img: np.ndarray,
    traj: np.ndarray,
    ctx: BEVPlanContext,
    result: Optional[BEVCheckResult] = None,
    color=(0, 255, 255),
) -> np.ndarray:
    """Draw a path on an unflipped BEV image whose pixels are grid cells
    (row 0 = z_min). Waypoints with a violation are drawn red."""
    bad = {v.waypoint for v in result.violations} if result is not None else set()
    pts = ego_to_cam_xz(np.asarray(traj))
    prev = None
    for i, (x, z) in enumerate(pts):
        cell = ctx.cell(x, z)
        if cell is None:
            prev = None
            continue
        p = (cell[1], cell[0])
        col = (0, 0, 255) if i in bad else color
        if prev is not None:
            cv2.line(bev_img, prev, p, col, 1, cv2.LINE_AA)
        prev = p
    return bev_img
