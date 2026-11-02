"""Ego-motion compensation.

Two backends:
  PoseEgoCompensation  — pose-based 3D (preferred for KITTI-360)
  estimate_homography  — ORB 2D fallback (kept for non-KITTI sequences)
"""
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np


def estimate_homography(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    max_features: int = 200,
) -> Optional[np.ndarray]:
    """
    Estimate 3×3 homography from background ORB matches between two frames.
    Returns None if too few inliers are found.
    frames must be RGB, shape (H,W,3) uint8 — the pipeline standard used throughout the tracker.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_RGB2GRAY)

    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn_matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        return None
    inliers = int(mask.sum()) if mask is not None else 0
    return H if inliers >= 15 else None


def compensate_centroids(
    centroids: list[list[float]],
    H: Optional[np.ndarray],
) -> list[list[float]]:
    """Apply homography H to a list of [x, y] pixel centroids."""
    if H is None or len(centroids) == 0:
        return [list(c) for c in centroids]

    pts = np.array(centroids, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2).tolist()


def _slerp_rotation(R0: np.ndarray, R1: np.ndarray, a: float) -> np.ndarray:
    """Interpolate between two rotation matrices (a=0 → R0, a=1 → R1)."""
    R_rel = R0.T @ R1
    rvec, _ = cv2.Rodrigues(R_rel)
    R_step, _ = cv2.Rodrigues(rvec * a)
    return R0 @ R_step


def interpolate_pose(poses: dict[int, np.ndarray], frame_idx: int) -> Optional[np.ndarray]:
    """Return the 4×4 pose for frame_idx, interpolating missing frames.

    KITTI-360 poses.txt does not list every frame (frames where the car is
    standing still, and some at the start, are left out). Missing frames inside
    the known range are interpolated (rotation SLERP + linear translation);
    frames outside it return None.
    """
    if frame_idx in poses:
        return poses[frame_idx]
    keys = sorted(poses)
    if not keys or frame_idx < keys[0] or frame_idx > keys[-1]:
        return None
    i = int(np.searchsorted(keys, frame_idx))
    f0, f1 = keys[i - 1], keys[i]
    a = (frame_idx - f0) / float(f1 - f0)
    P0, P1 = poses[f0], poses[f1]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _slerp_rotation(P0[:3, :3], P1[:3, :3], a)
    T[:3, 3] = (1.0 - a) * P0[:3, 3] + a * P1[:3, 3]
    return T


class PoseEgoCompensation:
    """Pose-based 3D ego-motion compensation for KITTI-360.

    Uses consecutive IMU/pose transforms to compute the ego displacement
    in rectified camera coordinates and applies it to Kalman filter states,
    so stationary world objects stay stationary in the tracker's 3D frame.

    Frames missing from poses.txt are interpolated, and the transform is
    always computed against the last frame that was actually processed, so
    skipped frames (e.g. --frame-stride) are handled correctly.

    Parameters
    ----------
    poses : dict[int, np.ndarray]
        {frame_idx: 4×4 T_imu_to_world} — loaded from poses.txt.
    T_pose_to_cam : np.ndarray
        4×4 transform from IMU/pose frame to unrectified camera frame.
    R_rect : np.ndarray
        3×3 rectification rotation.
    """

    def __init__(
        self,
        poses: dict[int, np.ndarray],
        T_pose_to_cam: np.ndarray,
        R_rect: np.ndarray,
    ) -> None:
        self._poses = poses
        self._T_pose_to_cam = np.asarray(T_pose_to_cam, dtype=np.float64)
        self._R_rect_4x4 = np.eye(4, dtype=np.float64)
        self._R_rect_4x4[:3, :3] = R_rect
        self._T_world_to_rectcam_prev: Optional[np.ndarray] = None

    def _world_to_rectcam(self, frame_idx: int) -> Optional[np.ndarray]:
        T_imu_to_world = interpolate_pose(self._poses, frame_idx)
        if T_imu_to_world is None:
            return None
        T_world_to_imu = np.linalg.inv(T_imu_to_world)
        return self._R_rect_4x4 @ self._T_pose_to_cam @ T_world_to_imu

    def rectcam_to_world(self, frame_idx: int) -> Optional[np.ndarray]:
        """4×4 transform from rectified camera frame at frame_idx to world."""
        T = self._world_to_rectcam(frame_idx)
        return None if T is None else np.linalg.inv(T)

    def get_transform(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return 4×4 T_ego that maps cam_{t-1} positions to cam_t.

        A stationary world point x in cam_{t-1} becomes T_ego @ x in cam_t.
        Returns None for the first frame or if no pose can be found.
        """
        T_curr = self._world_to_rectcam(frame_idx)
        if T_curr is None:
            # Keep the previous pose: the next frame with a pose is then
            # compensated against it instead of losing compensation too.
            return None

        T_ego = None
        if self._T_world_to_rectcam_prev is not None:
            # T_ego = T_world_to_cam_curr @ inv(T_world_to_cam_prev)
            T_ego = T_curr @ np.linalg.inv(self._T_world_to_rectcam_prev)

        self._T_world_to_rectcam_prev = T_curr
        return T_ego
