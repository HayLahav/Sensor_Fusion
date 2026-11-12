"""Camera projection operators.

A projection travels with the data as an object instead of a bare matrix, so
the geometry model (pinhole vs. fisheye) is explicit and a wrong argument
fails loudly. Idea borrowed from autowarefoundation/auto_e2e.

All points are in the rectified camera frame (x right, y down, z forward).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class CameraProjection(ABC):
    """Maps [N,3] camera-frame points to [N,2] pixels plus a validity mask."""

    @abstractmethod
    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (uv [N,2] float32, valid [N] bool)."""


class PinholeProjection(CameraProjection):
    def __init__(self, K: np.ndarray, min_depth: float = 0.1) -> None:
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape}")
        self.K = K
        self.min_depth = min_depth

    def project(self, pts_cam):
        pts = np.asarray(pts_cam, dtype=np.float32).reshape(-1, 3)
        valid = pts[:, 2] > self.min_depth
        z = np.where(valid, pts[:, 2], 1.0)
        uvw = (self.K @ pts.T).T
        uv = uvw[:, :2] / z[:, None]
        return uv.astype(np.float32), valid


class FisheyeProjection(CameraProjection):
    """Equidistant fisheye model: r = f * theta (f = mean of fx, fy)."""

    def __init__(self, K: np.ndarray, max_theta: float = np.pi / 2 * 0.98) -> None:
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape}")
        self.K = K
        self.max_theta = max_theta

    def project(self, pts_cam):
        pts = np.asarray(pts_cam, dtype=np.float32).reshape(-1, 3)
        rxy = np.hypot(pts[:, 0], pts[:, 1])
        theta = np.arctan2(rxy, pts[:, 2])
        valid = theta < self.max_theta
        scale = np.where(rxy > 1e-9, theta / np.maximum(rxy, 1e-9), 1.0 / np.maximum(pts[:, 2], 1e-9))
        f = 0.5 * (self.K[0, 0] + self.K[1, 1])
        u = f * pts[:, 0] * scale + self.K[0, 2]
        v = f * pts[:, 1] * scale + self.K[1, 2]
        return np.stack([u, v], axis=1).astype(np.float32), valid


def as_projection(cam) -> CameraProjection:
    """Accept a CameraProjection or a 3x3 intrinsics array (treated as pinhole)."""
    if isinstance(cam, CameraProjection):
        return cam
    return PinholeProjection(cam)
