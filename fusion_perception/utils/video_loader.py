"""Video frame iterator with metadata. Supports file paths and frame limits."""
from __future__ import annotations
import cv2
import numpy as np
from typing import Iterator, Optional
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("video_loader")


class VideoLoader:
    """Yields (frame_idx, frame_rgb, metadata) tuples from a video source."""

    def __init__(
        self,
        source: str,
        resize_hw: Optional[tuple[int, int]],
        max_frames: Optional[int],
    ) -> None:
        self.source = source
        self.resize_hw = resize_hw
        self.max_frames = max_frames

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.source}")
        return cap

    @property
    def fps(self) -> float:
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    @property
    def total_frames(self) -> int:
        cap = self._open()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, dict]]:
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        meta = {"fps": fps, "total_frames": total, "original_hw": (h, w)}
        logger.info(f"Opened {self.source}: {w}x{h} @ {fps:.1f}fps, {total} frames")

        frame_idx = 0
        while True:
            if self.max_frames is not None and frame_idx >= self.max_frames:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self.resize_hw is not None:
                th, tw = self.resize_hw
                frame_rgb = cv2.resize(frame_rgb, (tw, th))
            yield frame_idx, frame_rgb, meta
            frame_idx += 1

        cap.release()
        logger.info(f"VideoLoader finished: {frame_idx} frames yielded")
