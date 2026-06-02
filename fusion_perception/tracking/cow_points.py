"""CoWTracker point spawning and output unpacking for KalmanCoWTracker."""
from __future__ import annotations
import math
import numpy as np
import torch


def spawn_points(
    bbox2d: list[float],
    beta: float = 0.5,
    min_pts: int = 8,
    max_pts: int = 64,
) -> np.ndarray:
    """
    Sample a uniform grid of pixel points within bbox2d.
    n_points = clamp(int(β · sqrt(area)), min_pts, max_pts)
    Returns array of shape [N, 2] in pixel coords.
    """
    x1, y1, x2, y2 = bbox2d
    area = max(0.0, (x2 - x1) * (y2 - y1))
    n_raw = int(math.sqrt(area) * beta)
    side = max(1, int(math.ceil(math.sqrt(max(min_pts, min(max_pts, n_raw))))))
    # Use side×side so grid is always complete — then clamp
    n = min(side * side, max_pts)
    n = max(n, min_pts)

    xs = np.linspace(x1, x2, side, endpoint=False) + (x2 - x1) / (2 * side)
    ys = np.linspace(y1, y2, side, endpoint=False) + (y2 - y1) / (2 * side)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return pts[:n]


def unpack_cow_outputs(
    pred_tracks: torch.Tensor,   # [1, T, N_total, 2]
    pred_vis: torch.Tensor,      # [1, T, N_total]
    track_ids: list[int],
    point_counts: list[int],     # how many points each track contributed
    conf_threshold: float = 0.85,
    min_points: int = 4,
) -> tuple[dict[int, list[float]], dict[int, bool]]:
    """
    Split batched CoWTracker output back to per-track displacement and validity.

    Returns:
      displacements: {track_id: list[float]} — median pixel displacement over the current sliding window (not since track creation)
      valids: {track_id: bool} — True if ≥ min_points survived confidence gate
    """
    last_pos = pred_tracks[0, -1].cpu().numpy()   # [N_total, 2]
    first_pos = pred_tracks[0, 0].cpu().numpy()   # [N_total, 2]
    last_vis = pred_vis[0, -1].cpu().numpy()      # [N_total]

    displacements: dict[int, list[float]] = {}
    valids: dict[int, bool] = {}

    offset = 0
    for tid, n in zip(track_ids, point_counts):
        pts_disp = last_pos[offset:offset + n] - first_pos[offset:offset + n]
        pts_conf = last_vis[offset:offset + n]
        offset += n

        mask = pts_conf >= conf_threshold
        n_alive = int(mask.sum())
        if n_alive >= min_points:
            displacements[tid] = np.median(pts_disp[mask], axis=0).tolist()
            valids[tid] = True
        else:
            displacements[tid] = [0.0, 0.0]
            valids[tid] = False

    return displacements, valids
