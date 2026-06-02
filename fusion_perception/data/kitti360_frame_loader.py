"""KITTI-360 camera frame loader.

Reads rectified left camera images (image_00/data_rect/) as a video-like
iterator with the same interface as VideoLoader.

Frame indices are taken from the filename stems
(e.g. 0000000042.png → frame_idx=42) so they align with poses.txt and
GT label lookups in Kitti360GTLabels.
"""
from __future__ import annotations
import os
import numpy as np
import cv2
from glob import glob
from typing import Iterator, Optional
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("kitti360_frame_loader")

_KITTI360_FPS = 10.0


class Kitti360FrameLoader:
    """Iterates rectified camera frames from a KITTI-360 sequence.

    Yields (frame_idx: int, frame_rgb: np.ndarray [H,W,3] uint8, meta: dict).
    frame_idx matches the filename stem so it aligns with poses.txt / GT labels.
    """

    def __init__(
        self,
        dataset_root: str,
        sequence: str,
        resize_hw: Optional[tuple[int, int]] = None,
        max_frames: Optional[int] = None,
        camera: str = "image_00",
    ) -> None:
        self.dataset_root = dataset_root
        self.sequence = sequence
        self.resize_hw = resize_hw
        self.max_frames = max_frames
        self.camera = camera

        img_dir = os.path.join(
            dataset_root, "data_2d_raw", sequence, camera, "data_rect"
        )
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"KITTI-360 image dir not found: {img_dir}")
        self._files = sorted(glob(os.path.join(img_dir, "*.png")))
        if not self._files:
            raise FileNotFoundError(f"No PNG frames in {img_dir}")
        self._frame_indices = [
            int(os.path.splitext(os.path.basename(p))[0]) for p in self._files
        ]
        logger.info(
            f"Kitti360FrameLoader: {sequence}/{camera} | {len(self._files)} frames "
            f"(idx {self._frame_indices[0]}–{self._frame_indices[-1]})"
        )

    @property
    def fps(self) -> float:
        return _KITTI360_FPS

    @property
    def total_frames(self) -> int:
        n = len(self._files)
        return min(n, self.max_frames) if self.max_frames is not None else n

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, dict]]:
        meta = {
            "fps": _KITTI360_FPS,
            "total_frames": len(self._files),
            "dataset": "kitti360",
            "sequence": self.sequence,
        }
        for i, (path, frame_idx) in enumerate(zip(self._files, self._frame_indices)):
            if self.max_frames is not None and i >= self.max_frames:
                break
            bgr = cv2.imread(path)
            if bgr is None:
                logger.warning(f"Failed to read {path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.resize_hw is not None:
                th, tw = self.resize_hw
                rgb = cv2.resize(rgb, (tw, th))
            yield frame_idx, rgb, meta

    def get_intrinsics(self) -> np.ndarray:
        """Return 3×3 camera intrinsics K from calibration/perspective.txt."""
        calib_path = os.path.join(self.dataset_root, "calibration", "perspective.txt")
        try:
            from kitti360scripts.devkits.commons.loadCalibration import loadPerspectiveIntrinsic
            intrinsics = loadPerspectiveIntrinsic(calib_path)
            P = np.asarray(intrinsics["P_rect_00"], dtype=np.float32)
            return P[:3, :3]
        except ImportError:
            pass
        # Manual fallback
        with open(calib_path) as f:
            for line in f:
                if line.startswith("P_rect_00:"):
                    vals = [float(v) for v in line.split()[1:]]
                    P = np.array(vals, dtype=np.float32).reshape(3, 4)
                    return P[:3, :3]
        raise ValueError(f"P_rect_00 not found in {calib_path}")
