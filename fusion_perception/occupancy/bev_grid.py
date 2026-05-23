"""BEV occupancy grid with temporal exponential decay.

Grid convention:
  rows  = forward (z) axis, row 0 = z_min
  cols  = lateral (x) axis, col 0 = x_min
  value = occupancy probability [0.0, 1.0]

Cell value semantics:
  0.0–0.3  → free / decayed
  ~0.6     → LiDAR-observed, no tracked object
  1.0      → confirmed tracked object

TODO: Add ray-casting free-space estimation from ego origin.
TODO: Support multi-layer grids (height slices).
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from fusion_perception.utils.dataclasses import Track, OccupancyGrid
from fusion_perception.utils.geometry import world_to_grid
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("bev_grid")


class OccupancyBEVGenerator:
    """Stateful BEV occupancy grid. Call update() once per frame."""

    def __init__(
        self,
        resolution: float,
        x_range: list[float],
        z_range: list[float],
        decay_factor: float,
        lidar_confidence: float = 0.6,
    ) -> None:
        self.resolution = resolution
        self.x_range = x_range
        self.z_range = z_range
        self.decay_factor = decay_factor
        self.lidar_confidence = lidar_confidence

        n_rows = int((z_range[1] - z_range[0]) / resolution)
        n_cols = int((x_range[1] - x_range[0]) / resolution)
        self._grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    def update(
        self,
        tracks: list[Track],
        frame_idx: int,
        lidar_pts: Optional[np.ndarray] = None,
    ) -> OccupancyGrid:
        """Apply decay, rasterize LiDAR then tracks, return updated grid."""
        # 1. Decay
        self._grid *= self.decay_factor

        # 2. LiDAR pass — vectorized rasterization
        if lidar_pts is not None and len(lidar_pts) > 0:
            pts = np.asarray(lidar_pts, dtype=np.float32)
            xs, zs = pts[:, 0], pts[:, 2]
            in_range = (
                (xs >= self.x_range[0]) & (xs <= self.x_range[1]) &
                (zs >= self.z_range[0]) & (zs <= self.z_range[1])
            )
            xs, zs = xs[in_range], zs[in_range]
            cols = ((xs - self.x_range[0]) / self.resolution).astype(np.intp)
            rows = ((zs - self.z_range[0]) / self.resolution).astype(np.intp)
            rows = np.clip(rows, 0, self._grid.shape[0] - 1)
            cols = np.clip(cols, 0, self._grid.shape[1] - 1)
            self._grid[rows, cols] = np.maximum(
                self._grid[rows, cols], self.lidar_confidence
            )

        # 3. Track pass — confirmed objects always write 1.0
        n_rows, n_cols = self._grid.shape
        for track in tracks:
            if not track.position_3d_history:
                continue
            x, _, z = track.position_3d_history[-1]
            cell = world_to_grid(x, z, self.x_range, self.z_range, self.resolution)
            if cell is not None:
                row, col = cell
                if row < n_rows and col < n_cols:   # guard against boundary off-by-one
                    self._grid[row, col] = 1.0

        logger.debug(
            f"Frame {frame_idx}: "
            f"{int((self._grid > 0.5).sum())} occupied cells"
        )

        return OccupancyGrid(
            frame_idx=frame_idx,
            resolution=self.resolution,
            x_range=self.x_range,
            z_range=self.z_range,
            grid=self._grid.tolist(),
            decay_factor=self.decay_factor,
        )

    def get_freespace_mask(self) -> np.ndarray:
        """Binary mask: True = free cell."""
        return self._grid <= 0.5

    def reset(self) -> None:
        self._grid[:] = 0.0
        logger.info("OccupancyBEVGenerator reset")
