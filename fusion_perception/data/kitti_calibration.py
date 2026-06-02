"""KITTI calibration parsers — supports KITTI raw and KITTI-360.

Both expose a single method: velo_to_cam(pts) → rectified camera-frame points.

KITTI raw (seq_dir contains these files):
  calib_velo_to_cam.txt   — R (3×3) + T (3,)
  calib_cam_to_cam.txt    — R_rect_00 (3×3)

KITTI-360 (calib_dir = root/calibration/ subdir):
  calib_cam_to_velo.txt   — camera-00 → Velodyne rigid transform (kitti360scripts)
  perspective.txt          — R_rect_00, P_rect_00 intrinsics

kitti360scripts is used for KITTI-360 when available; falls back to manual
parsing of calib_cam_to_pose.txt + perspective.txt with velo≈pose approximation.
"""
from __future__ import annotations
import os
import numpy as np
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("kitti_calibration")


class KittiRawCalib:
    """Calibration for KITTI raw sequences."""

    def __init__(self, R_velo: np.ndarray, T_velo: np.ndarray, R_rect: np.ndarray) -> None:
        self._T = np.eye(4, dtype=np.float32)
        self._T[:3, :3] = R_velo
        self._T[:3, 3] = T_velo
        self._R_rect = R_rect.astype(np.float32)

    @classmethod
    def from_dir(cls, seq_dir: str) -> "KittiRawCalib":
        R, T = _parse_velo_to_cam(os.path.join(seq_dir, "calib_velo_to_cam.txt"))
        R_rect = _parse_r_rect_raw(os.path.join(seq_dir, "calib_cam_to_cam.txt"))
        logger.info("KittiRawCalib loaded")
        return cls(R, T, R_rect)

    def velo_to_cam(self, pts: np.ndarray) -> np.ndarray:
        """[N,3] LiDAR → rectified camera frame [N,3], front-facing only."""
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
        cam = (self._T @ pts_h.T).T[:, :3]
        rect = (self._R_rect @ cam.T).T
        return rect[rect[:, 2] > 0.1]


class Kitti360Calib:
    """Calibration for KITTI-360 sequences (image_00 / left perspective camera).

    Public attributes (always set, regardless of which path loaded the velo transform):
        T_pose_to_cam  : 4×4 float64 — IMU/pose frame → rectified camera frame
        R_rect         : 3×3 float64 — rectification rotation (R_rect_00)
    These are required by PoseEgoCompensation for ego-motion correction.
    """

    def __init__(
        self,
        T_velo_to_cam_rect: np.ndarray,
        T_pose_to_cam: np.ndarray,
        R_rect: np.ndarray,
    ) -> None:
        self._T = T_velo_to_cam_rect.astype(np.float32)
        self.T_pose_to_cam = np.asarray(T_pose_to_cam, dtype=np.float64)
        self.R_rect = np.asarray(R_rect, dtype=np.float64)

    @classmethod
    def from_dir(cls, calib_dir: str) -> "Kitti360Calib":
        # Always parse pose calibration — needed for ego-motion even in the velo path
        R_rect = _parse_r_rect_360(os.path.join(calib_dir, "perspective.txt"))
        T_cam_to_pose = _parse_cam_to_pose_360(
            os.path.join(calib_dir, "calib_cam_to_pose.txt")
        )
        T_pose_to_cam = np.linalg.inv(T_cam_to_pose.astype(np.float64))

        velo_file = os.path.join(calib_dir, "calib_cam_to_velo.txt")
        if os.path.exists(velo_file):
            return cls._from_kitti360scripts(calib_dir, velo_file, T_pose_to_cam, R_rect)
        return cls._from_fallback(T_pose_to_cam, R_rect)

    @classmethod
    def _from_kitti360scripts(
        cls,
        calib_dir: str,
        velo_file: str,
        T_pose_to_cam: np.ndarray,
        R_rect: np.ndarray,
    ) -> "Kitti360Calib":
        from kitti360scripts.devkits.commons.loadCalibration import (
            loadCalibrationRigid, loadPerspectiveIntrinsic,
        )
        T_cam_to_velo = loadCalibrationRigid(velo_file).astype(np.float32)
        T_velo_to_cam = np.linalg.inv(T_cam_to_velo)
        intrinsics = loadPerspectiveIntrinsic(os.path.join(calib_dir, "perspective.txt"))
        R_rect_f32 = intrinsics["R_rect_00"].astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_rect_f32 @ T_velo_to_cam[:3, :3]
        T[:3, 3] = R_rect_f32 @ T_velo_to_cam[:3, 3]
        logger.info("Kitti360Calib loaded via kitti360scripts (calib_cam_to_velo.txt)")
        return cls(T, T_pose_to_cam, R_rect)

    @classmethod
    def _from_fallback(
        cls,
        T_pose_to_cam: np.ndarray,
        R_rect: np.ndarray,
    ) -> "Kitti360Calib":
        """Fallback when calib_cam_to_velo.txt is absent (velo ≈ pose approximation)."""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_rect @ T_pose_to_cam[:3, :3]
        T[:3, 3] = R_rect @ T_pose_to_cam[:3, 3]
        logger.warning("Kitti360Calib: calib_cam_to_velo.txt not found, using velo≈pose approximation")
        return cls(T, T_pose_to_cam, R_rect)

    def velo_to_cam(self, pts: np.ndarray) -> np.ndarray:
        ones = np.ones((len(pts), 1), dtype=np.float32)
        pts_h = np.concatenate([pts.astype(np.float32), ones], axis=1)
        cam = (self._T @ pts_h.T).T[:, :3]
        return cam[cam[:, 2] > 0.1]


def load_calibration(root_dir: str) -> "KittiRawCalib | Kitti360Calib":
    """Auto-detect dataset format and return calibration object."""
    if os.path.exists(os.path.join(root_dir, "calib_velo_to_cam.txt")):
        return KittiRawCalib.from_dir(root_dir)
    calib360 = os.path.join(root_dir, "calibration")
    if os.path.isdir(calib360):
        return Kitti360Calib.from_dir(calib360)
    raise FileNotFoundError(
        f"No KITTI calibration found in {root_dir}. "
        "Expected calib_velo_to_cam.txt (raw) or calibration/ subdir (360)."
    )


# ── Private parsers ───────────────────────────────────────────────────────────

def _parse_velo_to_cam(path: str) -> tuple[np.ndarray, np.ndarray]:
    R = T = None
    with open(path) as f:
        for line in f:
            if line.startswith("R:"):
                R = np.array(line.split()[1:], dtype=np.float32).reshape(3, 3)
            elif line.startswith("T:"):
                T = np.array(line.split()[1:], dtype=np.float32)
    if R is None or T is None:
        raise ValueError(f"R or T missing in {path}")
    return R, T


def _parse_r_rect_raw(path: str) -> np.ndarray:
    with open(path) as f:
        for line in f:
            if line.startswith("R_rect_00:"):
                return np.array(line.split()[1:], dtype=np.float32).reshape(3, 3)
    return np.eye(3, dtype=np.float32)


def _parse_r_rect_360(path: str) -> np.ndarray:
    with open(path) as f:
        for line in f:
            if line.startswith("R_rect_00:"):
                vals = line.split()[1:]
                if len(vals) == 9:
                    return np.array(vals, dtype=np.float32).reshape(3, 3)
    return np.eye(3, dtype=np.float32)


def _parse_cam_to_pose_360(path: str) -> np.ndarray:
    """Parse image_00 entry from calib_cam_to_pose.txt → 4×4 matrix."""
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("image_00"):
            vals: list[str] = []
            for j in range(i + 1, min(i + 6, len(lines))):
                row = lines[j].strip()
                if row and not row[0].isalpha():
                    vals.extend(row.split())
                if len(vals) >= 12:
                    break
            if len(vals) >= 12:
                mat = np.array(vals[:12], dtype=np.float32).reshape(3, 4)
                T = np.eye(4, dtype=np.float32)
                T[:3, :] = mat
                return T
    logger.warning(f"image_00 not found in {path}, using identity")
    return np.eye(4, dtype=np.float32)
