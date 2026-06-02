"""BEV occupancy grid with temporal exponential decay, semantic labeling,
and ego-motion-compensated motion detection.

Motion grid:
  Computed as |current_occ - warp(prev_occ, T_ego)|.
  Because the warp perfectly compensates for ego motion using ground-truth
  KITTI-360 poses, only cells containing independently moving objects will
  have a high residual value.  Static background (road, parked cars, buildings)
  cancels out regardless of ego speed or turning.

Grid convention:
  rows  = forward (z) axis, row 0 = z_min
  cols  = lateral (x) axis, col 0 = x_min
"""
from __future__ import annotations
import math
import cv2
import numpy as np
from typing import Optional
from fusion_perception.utils.dataclasses import Track, OccupancyGrid
from fusion_perception.utils.geometry import world_to_grid
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("bev_grid")

_ROAD_SEM_CLASSES  = frozenset([0, 1, 9])
_MOTION_THRESHOLD  = 0.3   # occupancy delta above this = independent motion

_TRACK_SEM_CLASS: dict[str, int] = {
    "car": 13, "truck": 14, "bus": 15,
    "person": 11, "pedestrian": 11, "rider": 12,
    "cyclist": 18, "motorcycle": 17, "bicycle": 18,
}

_SEM_NAMES: dict[int, str] = {
    0: "road", 1: "sidewalk", 8: "vegetation", 9: "terrain",
    11: "person", 12: "rider", 13: "car", 14: "truck",
    15: "bus", 17: "motorcycle", 18: "cyclist",
}


def _build_sem_summary(sem_grid: np.ndarray, tracks: list[Track]) -> str:
    total = sem_grid.size
    if total == 0:
        return ""
    parts: list[str] = []
    for cls_id, name in _SEM_NAMES.items():
        count = int((sem_grid == cls_id).sum())
        if count > 0:
            parts.append(f"{name}:{100 * count // total}%")
    cls_counts: dict[str, int] = {}
    for t in tracks:
        cls_counts[t.class_name] = cls_counts.get(t.class_name, 0) + 1
    for cls, cnt in sorted(cls_counts.items()):
        parts.append(f"{cls}×{cnt}")
    return " ".join(parts)


def _compute_warp_matrix(
    T_ego: np.ndarray,
    x_range: list[float],
    z_range: list[float],
    resolution: float,
) -> np.ndarray:
    """Build a 2×3 affine matrix (grid-space) that maps BEV_{t-1} → BEV_t.

    Derived from the 3-D rigid transform T_ego (4×4, cam_{t-1}→cam_t):
      col' = a*col + b*row + col_offset
      row' = c*col + d*row + row_offset
    where a,b,c,d come from the x-z rows/cols of R, and the offset accounts
    for the non-zero grid origin (x_min, z_min ≠ 0).
    """
    R  = T_ego[:3, :3]
    t  = T_ego[:3, 3]
    a, b = float(R[0, 0]), float(R[0, 2])
    c, d = float(R[2, 0]), float(R[2, 2])
    tx_m, tz_m  = float(t[0]), float(t[2])
    x_min, z_min = x_range[0], z_range[0]
    res = resolution

    col_off = ((a - 1) * x_min + b * z_min + tx_m) / res
    row_off = (c * x_min + (d - 1) * z_min + tz_m) / res

    return np.float32([[a, b, col_off],
                       [c, d, row_off]])


class OccupancyBEVGenerator:
    """Stateful BEV occupancy + semantic + motion grid. Call update() once per frame."""

    def __init__(
        self,
        resolution: float,
        x_range: list[float],
        z_range: list[float],
        decay_factor: float,
        lidar_confidence: float = 0.6,
    ) -> None:
        self.resolution    = resolution
        self.x_range       = x_range
        self.z_range       = z_range
        self.decay_factor  = decay_factor
        self.lidar_confidence = lidar_confidence

        n_rows = int((z_range[1] - z_range[0]) / resolution)
        n_cols = int((x_range[1] - x_range[0]) / resolution)
        self._grid     = np.zeros((n_rows, n_cols), dtype=np.float32)
        self._sem_grid = np.full((n_rows, n_cols), -1, dtype=np.int16)
        self._prev_occ: Optional[np.ndarray] = None   # occupancy before decay, previous frame

    def update(
        self,
        tracks: list[Track],
        frame_idx: int,
        lidar_pts:  Optional[np.ndarray] = None,
        sem_mask:   Optional[np.ndarray] = None,
        intrinsics: Optional[np.ndarray] = None,
        T_ego:      Optional[np.ndarray] = None,
    ) -> OccupancyGrid:
        """Apply decay, rasterize LiDAR + tracks, compute motion grid."""

        # ── Save previous LiDAR-only occupancy for motion computation ────────
        # We use the pre-track snapshot (LiDAR only, no track overlay) so that
        # detection noise on track positions does not create false motion diffs.
        # Track cells are injected as 1.0 but can shift by ~0.5m between frames
        # due to KF noise, producing |1.0 - 0| = 1.0 diffs for parked cars.
        prev_occ = self._prev_occ  # may be None on first frame

        # ── 1. Decay + reset semantic ─────────────────────────────────────────
        self._grid     *= self.decay_factor
        self._sem_grid[:] = -1

        # ── 2. LiDAR rasterization ────────────────────────────────────────────
        if lidar_pts is not None and len(lidar_pts) > 0:
            pts = np.asarray(lidar_pts, dtype=np.float32)
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
            in_range = (
                (xs >= self.x_range[0]) & (xs <= self.x_range[1]) &
                (zs >= self.z_range[0]) & (zs <= self.z_range[1])
            )
            xs, ys, zs = xs[in_range], ys[in_range], zs[in_range]

            point_classes: Optional[np.ndarray] = None
            if sem_mask is not None and intrinsics is not None and len(xs) > 0:
                K   = intrinsics.astype(np.float32)
                uvz = (K @ np.stack([xs, ys, zs], axis=1).T).T
                us  = (uvz[:, 0] / uvz[:, 2]).astype(np.int32)
                vs  = (uvz[:, 1] / uvz[:, 2]).astype(np.int32)
                H_img, W_img = sem_mask.shape[:2]
                in_img = (us >= 0) & (us < W_img) & (vs >= 0) & (vs < H_img)
                raw_classes = np.full(len(xs), -1, dtype=np.int16)
                raw_classes[in_img] = sem_mask[vs[in_img], us[in_img]]
                point_classes = raw_classes

            if len(xs) > 0:
                cols = ((xs - self.x_range[0]) / self.resolution).astype(np.intp)
                rows = ((zs - self.z_range[0]) / self.resolution).astype(np.intp)
                rows = np.clip(rows, 0, self._grid.shape[0] - 1)
                cols = np.clip(cols, 0, self._grid.shape[1] - 1)
                self._grid[rows, cols] = np.maximum(
                    self._grid[rows, cols], self.lidar_confidence
                )
                if point_classes is not None:
                    self._sem_grid[rows, cols] = point_classes

        # ── 3. Capture LiDAR-only snapshot (before track overlay) ───────────────
        # Used for BOTH the motion diff this frame AND stored as prev_occ for
        # next frame.  Track cells (always 1.0) must be excluded from both sides
        # of the diff — otherwise a parked car gets:
        #   curr = 1.0 (track overlay)  vs  warped_prev = 0.6 (LiDAR only)
        #   diff = 0.4 > 0.3 → falsely "moving"
        lidar_only = self._grid.copy()

        # ── 4. Track rasterization (written into self._grid, NOT lidar_only) ──
        n_rows, n_cols = self._grid.shape
        for track in tracks:
            if not track.position_3d_history:
                continue
            x, _, z = track.position_3d_history[-1]
            cell = world_to_grid(x, z, self.x_range, self.z_range, self.resolution)
            if cell is not None:
                row, col = cell
                if row < n_rows and col < n_cols:
                    self._grid[row, col] = 1.0
                    sem_cls = _TRACK_SEM_CLASS.get(track.class_name, -1)
                    self._sem_grid[row, col] = sem_cls

        # ── 5. Motion grid via BEV warp-and-diff (LiDAR-only on both sides) ──
        # lidar_only  = current LiDAR returns (~0.6 at occupied cells)
        # warped      = previous LiDAR returns warped by T_ego (~0.6 at same cells)
        # diff ≈ 0 for static objects, diff ≈ 0.6 for truly moving objects
        motion_grid_np: Optional[np.ndarray] = None
        if T_ego is not None and prev_occ is not None:
            try:
                M = _compute_warp_matrix(T_ego, self.x_range, self.z_range, self.resolution)
                warped = cv2.warpAffine(
                    prev_occ, M, (n_cols, n_rows),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )
                diff = np.abs(lidar_only - warped)
                # Suppress cells empty in both frames (sensor noise on free space)
                occupied = (lidar_only > 0.25) | (warped > 0.25)
                diff[~occupied] = 0.0
                motion_grid_np = np.clip(diff, 0.0, 1.0).astype(np.float32)
            except Exception as exc:
                logger.debug(f"Motion grid computation failed: {exc}")

        # ── 6. Store LiDAR-only snapshot for next frame ───────────────────────
        self._prev_occ = lidar_only

        sem_summary = _build_sem_summary(self._sem_grid, tracks)
        logger.debug(f"Frame {frame_idx}: {int((self._grid > 0.5).sum())} occupied cells")

        return OccupancyGrid(
            frame_idx=frame_idx,
            resolution=self.resolution,
            x_range=self.x_range,
            z_range=self.z_range,
            grid=self._grid.tolist(),
            decay_factor=self.decay_factor,
            sem_grid=self._sem_grid.tolist(),
            sem_summary=sem_summary,
            motion_grid=motion_grid_np.tolist() if motion_grid_np is not None else None,
        )

    def get_freespace_mask(self) -> np.ndarray:
        return self._grid <= 0.5

    def reset(self) -> None:
        self._grid[:]      = 0.0
        self._sem_grid[:]  = -1
        self._prev_occ     = None
        logger.info("OccupancyBEVGenerator reset")
