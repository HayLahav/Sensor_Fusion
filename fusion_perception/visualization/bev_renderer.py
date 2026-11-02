"""Render BEV occupancy/semantic grid as a top-down image with track annotations.

Three per-track overlays (all drawn pre-flip in native grid resolution):
  1. Footprint  — rotated rectangle (w × l at yaw ry) replacing the centroid dot;
                  parked tracks (motion gate "static") are drawn grey and hollow
  2. Velocity   — ego-compensated arrow, length ∝ speed (moving tracks only)
  3. Prediction — ghost dot + dashed line at pos + velocity × T_PRED seconds

Optional BEVOverlay layers (render_bev(..., overlay=...)):
  - LiDAR map underlay: accumulated world map around the car (faint grey)
  - obstacles: ground-removed LiDAR + parked boxes used by the BEV check (white)
  - driven path (green) and planner paths (cyan), waypoints flagged by the
    BEV check in red
  - a one-line legend under the image
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import OccupancyGrid, Track
from fusion_perception.utils.geometry import world_to_grid

_T_PRED   = 1.5   # seconds ahead for predicted position
_T_ARROW  = 1.0   # seconds used to scale velocity arrow length
_SPEED_MIN = 1.0  # m/s — below this a track is treated as stationary

# Cityscapes trainId → BGR color
_SEM_BGR: dict[int, tuple[int, int, int]] = {
    -1: ( 40,  40,  40),
     0: (128,  64, 128),   # road
     1: (232,  35, 244),   # sidewalk
     2: ( 70,  70,  70),   # building
     3: (100, 100, 156),
     4: (153, 153, 190),
     5: (153, 153, 153),
     6: ( 30, 170, 250),
     7: (  0, 220, 220),
     8: ( 35, 142, 107),   # vegetation
     9: (152, 251, 152),   # terrain
    10: (180, 130,  70),
    11: ( 60,  20, 220),   # person
    12: (  0,   0, 255),   # rider
    13: (142,   0,   0),   # car
    14: ( 70,   0,   0),   # truck
    15: (100,  60,   0),   # bus
    16: (100,  80,   0),
    17: (230,   0,   0),   # motorcycle
    18: ( 32,  11, 119),   # bicycle
}
_DEFAULT_BGR = (80, 80, 80)

_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72),  (23, 204, 146), (134, 219, 61),
    (52, 147, 26),  (187, 212, 0),  (168, 153, 44), (255, 194, 0),
    (147, 69, 52),  (115, 100, 100),(236, 24, 0),   (255, 56, 132),
    (133, 0, 82),   (255, 56, 203), (200, 149, 255),(199, 55, 255),
]


def _track_color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _sem_color(cls_id: int) -> tuple[int, int, int]:
    return _SEM_BGR.get(int(cls_id), _DEFAULT_BGR)


def _faded(color: tuple[int, int, int], factor: float = 0.45) -> tuple[int, int, int]:
    return tuple(max(0, int(c * factor)) for c in color)  # type: ignore[return-value]


@dataclass
class BEVOverlay:
    """Extra layers for render_bev. Paths are in the ego frame (x forward,
    y left, heading), as produced by fusion_perception.planning."""
    map_points_xz: Optional[np.ndarray] = None          # [N,2] camera x, z
    obstacle: Optional[np.ndarray] = None               # bool [rows, cols]
    driven_path: Optional[np.ndarray] = None            # [N,3]
    driven_bad: set = field(default_factory=set)        # waypoint indices
    planner_paths: list = field(default_factory=list)   # [[N,3], ...]
    planner_bad: list = field(default_factory=list)     # [set, ...] per path
    planner_label: str = "plan"


_C_DRIVEN = (80, 220, 80)
_C_PLAN = (255, 220, 0)
_C_BAD = (40, 40, 255)
_C_OBST = (235, 235, 235)
_C_MAP = (95, 95, 95)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    dash: int = 4,
    gap: int = 3,
) -> None:
    x1, y1 = pt1; x2, y2 = pt2
    dx, dy = x2 - x1, y2 - y1
    dist = max(1, int(math.hypot(dx, dy)))
    step = dash + gap
    t = 0
    while t < dist:
        t0 = t / dist
        t1 = min((t + dash) / dist, 1.0)
        p0 = (int(x1 + t0 * dx), int(y1 + t0 * dy))
        p1 = (int(x1 + t1 * dx), int(y1 + t1 * dy))
        cv2.line(img, p0, p1, color, 1, cv2.LINE_AA)
        t += step


def _world_corners_to_grid(
    cx: float, cz: float,
    w: float, l: float, ry: float,
    x_range: list[float], z_range: list[float], resolution: float,
) -> np.ndarray | None:
    """Return [4,2] int32 (col,row) array for the BEV footprint, or None."""
    half_w, half_l = w / 2.0, l / 2.0
    local = np.array([
        [-half_w, -half_l],
        [ half_w, -half_l],
        [ half_w,  half_l],
        [-half_w,  half_l],
    ], dtype=np.float32)
    cos_r, sin_r = math.cos(ry), math.sin(ry)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
    world_xy = (R @ local.T).T + np.array([cx, cz], dtype=np.float32)

    pts: list[list[int]] = []
    for wx, wz in world_xy:
        cell = world_to_grid(float(wx), float(wz), x_range, z_range, resolution)
        if cell is None:
            return None
        pts.append([cell[1], cell[0]])   # (col, row)
    return np.array(pts, dtype=np.int32)


# ── Per-track overlay ─────────────────────────────────────────────────────────

def _draw_track(
    img: np.ndarray,
    track: Track,
    grid: OccupancyGrid,
) -> None:
    """Draw footprint, velocity arrow, and (if moving) predicted position."""
    if not track.position_3d_history:
        return

    color   = _track_color(track.track_id)
    x, _, z = track.position_3d_history[-1]
    cell    = world_to_grid(x, z, grid.x_range, grid.z_range, grid.resolution)
    if cell is None:
        return
    row, col = cell
    res = grid.resolution

    vx, _, vz = track.velocity_3d
    speed = math.hypot(vx, vz)

    # Prefer the tracker's motion gate. Otherwise use motion_grid (BEV
    # warp-and-diff), falling back to a KF speed threshold when the motion
    # grid is not yet available (first frame or no pose data).
    motion_val = 0.0
    state = getattr(track, "motion_state", "unknown")
    if state == "static":
        color = (150, 150, 150)
    if state in ("static", "moving"):
        moving = state == "moving"
    elif grid.motion_grid is not None:
        cell_mg = world_to_grid(x, z, grid.x_range, grid.z_range, grid.resolution)
        if cell_mg is not None:
            try:
                motion_val = float(grid.motion_grid[cell_mg[0]][cell_mg[1]])
            except (IndexError, TypeError):
                motion_val = 0.0
        moving = motion_val >= 0.3
    else:
        moving = speed >= _SPEED_MIN

    # ── 1. Footprint (rotated rectangle) ─────────────────────────────────────
    if track.last_box_3d is not None:
        cx_, cy_, cz_, w, h, l, ry = track.last_box_3d
        corners = _world_corners_to_grid(
            cx_, cz_, max(w, 0.3), max(l, 0.3), ry,
            grid.x_range, grid.z_range, res,
        )
        if corners is not None:
            if state != "static":            # parked: outline only
                overlay = img.copy()
                cv2.fillPoly(overlay, [corners], color)
                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            cv2.polylines(img, [corners], True, color, 1, cv2.LINE_AA)
    else:
        # Fallback: plain circle when no 3D box is available
        cv2.circle(img, (col, row), 4, color, -1)

    # White centre dot so the track ID reference point is always visible
    cv2.circle(img, (col, row), 2, (255, 255, 255), -1)

    # ── 2. Velocity arrow (ego-compensated, only for moving tracks) ───────────
    if moving:
        d_col = int(vx / res * _T_ARROW)
        d_row = int(vz / res * _T_ARROW)
        tip = (col + d_col, row + d_row)
        cv2.arrowedLine(img, (col, row), tip, (255, 255, 255), 2,
                        cv2.LINE_AA, tipLength=0.3)

    # ── 3. Predicted position (dashed line + ghost) ───────────────────────────
    if moving:
        pred_x = x + vx * _T_PRED
        pred_z = z + vz * _T_PRED
        pred_cell = world_to_grid(
            pred_x, pred_z, grid.x_range, grid.z_range, res
        )
        if pred_cell is not None:
            pred_row, pred_col = pred_cell
            ghost = _faded(color, 0.55)
            _dashed_line(img, (col, row), (pred_col, pred_row), ghost)
            cv2.circle(img, (pred_col, pred_row), 4, ghost, 1, cv2.LINE_AA)


# ── Main render functions ─────────────────────────────────────────────────────

def _ego_path_cells(path: np.ndarray, grid: OccupancyGrid) -> list[Optional[tuple[int, int]]]:
    out = []
    for x_e, y_e in np.asarray(path)[:, :2]:
        out.append(world_to_grid(float(-y_e), float(x_e), grid.x_range, grid.z_range,
                                 grid.resolution))
    return out


def _draw_path(img, path, grid, color, bad: set, thickness=1) -> None:
    cells = _ego_path_cells(path, grid)
    prev = (img.shape[1] // 2, _ego_row_unflipped(grid, img.shape[0]))   # start at the ego
    for i, cell in enumerate(cells):
        if cell is None:
            prev = None
            continue
        p = (cell[1], cell[0])
        if prev is not None:
            cv2.line(img, prev, p, _C_BAD if i in bad else color, thickness, cv2.LINE_AA)
        prev = p


def _ego_row_unflipped(grid: OccupancyGrid, n_rows: int) -> int:
    return min(n_rows - 1, max(0, int(-grid.z_range[0] / grid.resolution)))


def _apply_underlays(img: np.ndarray, grid: OccupancyGrid,
                     overlay: Optional[BEVOverlay], background_mask: np.ndarray) -> None:
    """LiDAR map dots on cells with no semantic class, then obstacle tint."""
    if overlay is None:
        return
    n_rows, n_cols = img.shape[:2]
    if overlay.map_points_xz is not None and len(overlay.map_points_xz):
        xz = np.asarray(overlay.map_points_xz)
        c = ((xz[:, 0] - grid.x_range[0]) / grid.resolution).astype(np.intp)
        r = ((xz[:, 1] - grid.z_range[0]) / grid.resolution).astype(np.intp)
        ok = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
        r, c = r[ok], c[ok]
        bg = background_mask[r, c]
        img[r[bg], c[bg]] = _C_MAP
    if overlay.obstacle is not None and overlay.obstacle.shape == (n_rows, n_cols):
        tint = img.copy()
        tint[overlay.obstacle] = _C_OBST
        cv2.addWeighted(tint, 0.6, img, 0.4, 0, img)


def _apply_paths(img: np.ndarray, grid: OccupancyGrid, overlay: Optional[BEVOverlay]) -> None:
    if overlay is None:
        return
    if overlay.driven_path is not None and len(overlay.driven_path):
        _draw_path(img, overlay.driven_path, grid, _C_DRIVEN, overlay.driven_bad)
    for i, path in enumerate(overlay.planner_paths):
        bad = overlay.planner_bad[i] if i < len(overlay.planner_bad) else set()
        _draw_path(img, path, grid, _C_PLAN, bad)


def _legend(img: np.ndarray, overlay: Optional[BEVOverlay]) -> np.ndarray:
    if overlay is None:
        return img
    items = []
    if overlay.driven_path is not None:
        items.append(("driven", _C_DRIVEN))
    if overlay.planner_paths:
        items.append((overlay.planner_label, _C_PLAN))
    if overlay.driven_bad or any(overlay.planner_bad):
        items.append(("flagged", _C_BAD))
    if overlay.obstacle is not None:
        items.append(("obstacle", _C_OBST))
    items.append(("parked", (150, 150, 150)))
    bar = np.zeros((18, img.shape[1], 3), np.uint8)
    x = 4
    for text, col in items:
        cv2.rectangle(bar, (x, 5), (x + 8, 13), col, -1)
        cv2.putText(bar, text, (x + 11, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1,
                    cv2.LINE_AA)
        x += 16 + 7 * len(text)
    return np.vstack([img, bar])


def _finish(bev_img: np.ndarray, grid: OccupancyGrid, size_px: int,
            overlay: Optional[BEVOverlay]) -> np.ndarray:
    n_rows, n_cols = bev_img.shape[:2]
    bev_img = cv2.flip(bev_img, 0)
    ego_col = n_cols // 2
    ego_row = n_rows - 1 - _ego_row_unflipped(grid, n_rows)
    cv2.circle(bev_img, (ego_col, ego_row), 5, (0, 255, 0), -1)
    cv2.circle(bev_img, (ego_col, ego_row), 5, (0, 0, 0), 1)
    out = cv2.resize(bev_img, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    if overlay is None:
        return out
    # Keep the square size the compositor expects: drop the farthest rows for the legend
    with_legend = _legend(out, overlay)
    return with_legend[-size_px:] if with_legend.shape[0] > size_px else with_legend


def _render_semantic(
    sem_grid: np.ndarray,
    tracks: list[Track],
    grid: OccupancyGrid,
    size_px: int,
    overlay: Optional[BEVOverlay] = None,
) -> np.ndarray:
    n_rows, n_cols = sem_grid.shape
    bev_img = np.full((n_rows, n_cols, 3), _SEM_BGR[-1], dtype=np.uint8)

    unique_cls = np.unique(sem_grid)
    for cls_id in unique_cls:
        bev_img[sem_grid == cls_id] = _sem_color(cls_id)

    _apply_underlays(bev_img, grid, overlay, background_mask=(sem_grid < 0))
    for track in tracks:
        _draw_track(bev_img, track, grid)
    _apply_paths(bev_img, grid, overlay)
    return _finish(bev_img, grid, size_px, overlay)


def render_bev(
    grid: OccupancyGrid,
    tracks: list[Track],
    size_px: int = 400,
    overlay: Optional[BEVOverlay] = None,
) -> np.ndarray:
    """Render BEV as [size_px, size_px, 3] BGR image."""
    rows = len(grid.grid)
    cols = len(grid.grid[0]) if rows > 0 else 1

    if grid.sem_grid is not None:
        return _render_semantic(
            np.array(grid.sem_grid, dtype=np.int16), tracks, grid, size_px, overlay
        )

    # ── Legacy occupancy fallback ─────────────────────────────────────────────
    occupancy = np.array(grid.grid, dtype=np.float32)
    bev_img   = np.zeros((rows, cols, 3), dtype=np.uint8)

    bev_img[:, :, 1] = np.where(occupancy < 0.3, 35, 0).astype(np.uint8)
    lidar = (occupancy >= 0.3) & (occupancy < 0.8)
    bev_img[:, :, 0] = np.where(lidar, (occupancy * 200).astype(np.uint8), 0)
    hi = occupancy >= 0.8
    bev_img[:, :, 2] = np.where(hi, (occupancy * 255).astype(np.uint8), 0)
    bev_img[:, :, 1] = np.where(hi, (occupancy * 80).astype(np.uint8), bev_img[:, :, 1])

    _apply_underlays(bev_img, grid, overlay, background_mask=(occupancy < 0.3))
    for track in tracks:
        _draw_track(bev_img, track, grid)
    _apply_paths(bev_img, grid, overlay)
    return _finish(bev_img, grid, size_px, overlay)
