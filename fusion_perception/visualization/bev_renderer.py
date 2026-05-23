"""Render BEV occupancy grid as a top-down image with track positions."""
from __future__ import annotations
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import OccupancyGrid, Track
from fusion_perception.utils.geometry import world_to_grid


def render_bev(
    grid: OccupancyGrid,
    tracks: list[Track],
    size_px: int = 400,
) -> np.ndarray:
    """Render occupancy grid and track positions as a BEV image [size_px, size_px, 3]."""
    rows = len(grid.grid)
    cols = len(grid.grid[0]) if rows > 0 else 1

    occupancy = np.array(grid.grid, dtype=np.float32)
    bev_img = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Map occupancy to red channel; free space as dark green
    bev_img[:, :, 2] = (occupancy * 255).astype(np.uint8)
    bev_img[:, :, 1] = ((1.0 - occupancy) * 40).astype(np.uint8)

    # Draw track centroids as colored dots
    for track in tracks:
        if not track.position_3d_history:
            continue
        x, _, z = track.position_3d_history[-1]
        cell = world_to_grid(x, z, grid.x_range, grid.z_range, grid.resolution)
        if cell:
            row, col = cell
            cv2.circle(bev_img, (col, row), 3, (255, 200, 0), -1)

            if len(track.position_3d_history) >= 2:
                px, _, pz = track.position_3d_history[-2]
                prev_cell = world_to_grid(px, pz, grid.x_range, grid.z_range, grid.resolution)
                if prev_cell:
                    pr, pc = prev_cell
                    cv2.arrowedLine(bev_img, (pc, pr), (col, row), (0, 255, 255), 1, tipLength=0.4)

    # Flip vertically: row 0 = z_min (near), want near at bottom
    bev_img = cv2.flip(bev_img, 0)
    ego_col = cols // 2
    cv2.circle(bev_img, (ego_col, rows - 2), 4, (0, 255, 0), -1)

    return cv2.resize(bev_img, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
