"""Decide whether a track is static or moving from its world-frame positions.

Fixes false motion on parked cars that suddenly appear. A new detection's
first positions are noisy (truncated box at the image edge, monocular depth),
and the Kalman filter turns that noise into a velocity. Instead, the tracker
keeps the measured positions of each track, warps them with the same ego
transform as the Kalman state (so they stay in the current camera frame and a
parked car's positions pile up in one place), and fits a straight line through
them over time. The slope is the object's own speed, independent of ego motion.

Until there are enough observations the state is "unknown" and the tracker
holds velocity at zero. LiDAR-based positions (median of the object's LiDAR
points, see utils/lidar_object.py) are much less noisy than box centres, so
they need fewer observations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

UNKNOWN = "unknown"
STATIC = "static"
MOVING = "moving"


@dataclass
class MotionObservation:
    frame_idx: int
    position: np.ndarray   # [x, y, z] in the current rectified camera frame
    source: str            # "lidar" | "box"


def warp_observations(obs: list[MotionObservation], T_ego: np.ndarray) -> None:
    """Move stored positions from cam_{t-1} into cam_t (in place)."""
    R = T_ego[:3, :3]
    t = T_ego[:3, 3]
    for o in obs:
        o.position = R @ o.position + t


def ground_speed(obs: list[MotionObservation], fps: float) -> Optional[float]:
    """Least-squares speed (m/s) in the ground plane (camera x, z)."""
    if len(obs) < 2:
        return None
    t = np.array([o.frame_idx for o in obs], dtype=np.float64) / max(fps, 1e-6)
    if np.ptp(t) <= 0:
        return None
    xz = np.array([[o.position[0], o.position[2]] for o in obs], dtype=np.float64)
    t_c = t - t.mean()
    slope = (t_c[:, None] * (xz - xz.mean(axis=0))).sum(axis=0) / (t_c ** 2).sum()
    return float(np.hypot(slope[0], slope[1]))


def classify_motion(
    obs: list[MotionObservation],
    fps: float,
    min_obs_lidar: int = 5,
    min_obs_box: int = 10,
    static_speed: float = 1.0,
) -> tuple[str, Optional[float]]:
    """Return (state, speed). Prefers LiDAR observations when enough exist."""
    lidar = [o for o in obs if o.source == "lidar"]
    box = [o for o in obs if o.source == "box"]
    if len(lidar) >= min_obs_lidar:
        use = lidar
    elif len(box) >= min_obs_box:
        use = box
    else:
        return UNKNOWN, None
    speed = ground_speed(use, fps)
    if speed is None:
        return UNKNOWN, None
    return (STATIC if speed < static_speed else MOVING), speed
