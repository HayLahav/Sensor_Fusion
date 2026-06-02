"""Dense 3D reconstruction via depth-map unprojection.

Accumulates a rolling world-frame point cloud from per-frame depth maps.
Useful for dense BEV background occupancy and scene mapping.

Each frame's depth map is unprojected using the camera intrinsics and
optionally transformed to world frame via the ego pose (T_cam_to_world).
When no pose is provided the cloud is kept in camera frame (useful for
single-frame analysis without GPS/IMU).

The rolling buffer retains the last `rolling_frames` clouds.  Call
`get_points()` to obtain all accumulated points as [N, 3] float32.

Down-sampling:
  Full depth maps at 1242×375 contain ~465k points per frame.  At 10 fps
  that is 4.65M points for a 10-frame window — too large for real-time BEV.
  `stride` subsamples the depth grid (default 4 → ~29k points/frame).
"""
from __future__ import annotations
from collections import deque
import numpy as np
from typing import Optional
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("depth.reconstruction")


def _unproject(
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    stride: int = 4,
    max_depth: float = 60.0,
) -> np.ndarray:
    """Unproject a depth map to camera-frame 3D points.

    Parameters
    ----------
    depth_map : np.ndarray
        [H, W] float32 depth in metres.
    intrinsics : np.ndarray
        [3, 3] camera intrinsics matrix K.
    stride : int
        Pixel stride for subsampling (reduces output size).
    max_depth : float
        Ignore pixels deeper than this.

    Returns
    -------
    np.ndarray
        [N, 3] float32 points in camera frame.
    """
    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_map[vs, us]
    valid = (z > 0.1) & (z < max_depth)

    z = z[valid].astype(np.float32)
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1)


class PointCloudAccumulator:
    """Rolling world-frame point cloud from monocular depth maps.

    Parameters
    ----------
    rolling_frames : int
        How many past frames to retain in the buffer.
    stride : int
        Pixel stride for depth-map subsampling.
    max_depth : float
        Points deeper than this (metres) are discarded.
    """

    def __init__(
        self,
        rolling_frames: int = 10,
        stride: int = 4,
        max_depth: float = 60.0,
    ) -> None:
        self.rolling_frames = rolling_frames
        self.stride = stride
        self.max_depth = max_depth
        self._clouds: deque[np.ndarray] = deque(maxlen=rolling_frames)

    def update(
        self,
        depth_map: np.ndarray,
        intrinsics: np.ndarray,
        T_cam_to_world: Optional[np.ndarray] = None,
    ) -> None:
        """Add one depth map's worth of 3D points to the buffer.

        Parameters
        ----------
        depth_map : np.ndarray
            [H, W] float32 metric depth.
        intrinsics : np.ndarray
            [3, 3] camera intrinsics K.
        T_cam_to_world : np.ndarray | None
            4×4 rigid transform from camera to world frame.
            Pass None to accumulate in camera frame (useful without pose).
        """
        pts_cam = _unproject(depth_map, intrinsics, self.stride, self.max_depth)
        if len(pts_cam) == 0:
            return

        if T_cam_to_world is not None:
            R = T_cam_to_world[:3, :3].astype(np.float32)
            t = T_cam_to_world[:3, 3].astype(np.float32)
            pts_world = (R @ pts_cam.T).T + t
        else:
            pts_world = pts_cam

        self._clouds.append(pts_world)
        logger.debug(
            f"PointCloudAccumulator: added {len(pts_world)} pts "
            f"(buffer={len(self._clouds)}/{self.rolling_frames})"
        )

    def get_points(self) -> np.ndarray:
        """Return all accumulated points as [N, 3] float32.

        Returns an empty [0, 3] array if the buffer is empty.
        """
        if not self._clouds:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(list(self._clouds), axis=0)

    def clear(self) -> None:
        self._clouds.clear()
