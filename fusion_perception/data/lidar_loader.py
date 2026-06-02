"""KITTI LiDAR .bin loader — supports KITTI raw and KITTI-360.

Usage:
    seq = KittiRawLidar.from_dir('/path/to/seq')          # raw
    seq = Kitti360Lidar.from_dir('/path/to/drive')        # 360
    path = seq.get_path(frame_idx)                        # None if missing
    pts  = load_bin(path)                                 # [N, 3] float32
"""
from __future__ import annotations
import os
import numpy as np
from glob import glob
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("lidar_loader")


def load_bin(path: str) -> np.ndarray:
    """Load a KITTI .bin file → [N, 3] float32 (intensity column dropped)."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]


def load_bin_4col(path: str) -> np.ndarray:
    """Load a KITTI .bin file → [N, 4] float32 (x, y, z, intensity)."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


class KittiRawLidar:
    """Sorted .bin file list for a KITTI raw sequence."""

    def __init__(self, files: list[str]) -> None:
        self._files = files
        logger.info(f"KittiRawLidar: {len(files)} frames")

    @classmethod
    def from_dir(cls, seq_dir: str) -> "KittiRawLidar":
        """seq_dir: root of KITTI raw sequence (contains velodyne_points/)."""
        velo_dir = os.path.join(seq_dir, "velodyne_points", "data")
        files = sorted(glob(os.path.join(velo_dir, "*.bin")))
        if not files:
            raise FileNotFoundError(f"No .bin files in {velo_dir}")
        return cls(files)

    def get_path(self, frame_idx: int) -> str | None:
        if 0 <= frame_idx < len(self._files):
            return self._files[frame_idx]
        return None


class Kitti360Lidar:
    """Sorted .bin file list for a KITTI-360 drive."""

    def __init__(self, files: list[str]) -> None:
        self._files = files
        # filename stem → path (e.g. 0000000042.bin → 42) for frame-aligned lookup
        self._by_frame: dict[int, str] = {
            int(os.path.splitext(os.path.basename(p))[0]): p for p in files
        }
        logger.info(f"Kitti360Lidar: {len(files)} frames")

    @classmethod
    def from_dir(cls, drive_dir: str) -> "Kitti360Lidar":
        """drive_dir: KITTI-360 drive root or data_3d_raw parent.

        Tries these paths in order:
          {drive_dir}/velodyne_points/data/
          {drive_dir}/data_3d_raw/*/velodyne_points/data/
        """
        candidate = os.path.join(drive_dir, "velodyne_points", "data")
        if os.path.isdir(candidate):
            velo_dir = candidate
        else:
            matches = glob(os.path.join(
                drive_dir, "data_3d_raw", "*", "velodyne_points", "data"
            ))
            if not matches:
                raise FileNotFoundError(
                    f"velodyne_points/data not found under {drive_dir}"
                )
            velo_dir = matches[0]
        files = sorted(glob(os.path.join(velo_dir, "*.bin")))
        if not files:
            raise FileNotFoundError(f"No .bin files in {velo_dir}")
        return cls(files)

    def get_path(self, frame_idx: int) -> str | None:
        # Prefer exact frame-number lookup (aligns with Kitti360FrameLoader indices)
        if frame_idx in self._by_frame:
            return self._by_frame[frame_idx]
        # Fallback: positional lookup
        if 0 <= frame_idx < len(self._files):
            return self._files[frame_idx]
        return None
