"""Thin wrapper around py123d SceneAPI.

Yields (frame_idx, frame_rgb, meta) identical to VideoLoader so
StreamingPipeline._run_frame_loop() works with either source.

py123d API calls are isolated here — if py123d's API changes,
only this file needs updating. The three extra methods expose
calibration, LiDAR, and GT labels for benchmark-aware runs.

NOTE: py123d must be installed and logs pre-converted via:
  py123d-conversion dataset=<name> ...
before constructing this loader.
"""
from __future__ import annotations
import numpy as np
from typing import Iterator, Optional

try:
    from py123d import SceneAPI
except ImportError:
    SceneAPI = None  # type: ignore[assignment,misc]

from fusion_perception.utils.dataclasses import GTLabel
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("py123d_loader")


class Py123dLoader:
    """Iterates camera frames from a py123d-converted log.

    Yields: (frame_idx: int, frame_rgb: np.ndarray [H,W,3], meta: dict)
    """

    def __init__(self, log_dir: str, camera_name: str = "camera") -> None:
        if SceneAPI is None:
            raise ImportError(
                "py123d not installed. Run: pip install py123d[kitti-360]"
            )
        self._scene = SceneAPI(log_dir)
        self._camera_name = camera_name
        self._cam = self._scene.cameras[camera_name]
        self._frames: list[tuple[float, np.ndarray]] = list(self._cam.frames)
        self._fps: float = float(self._scene.fps)
        logger.info(
            f"Py123dLoader: {log_dir} | camera={camera_name} "
            f"| {len(self._frames)} frames @ {self._fps:.1f}fps"
        )

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, dict]]:
        for frame_idx, (ts, frame_rgb) in enumerate(self._frames):
            yield frame_idx, frame_rgb, {
                "fps": self._fps,
                "total_frames": len(self._frames),
                "timestamp": ts,
                "camera_name": self._camera_name,
            }

    def get_intrinsics(self) -> np.ndarray:
        """Return real camera intrinsics [3,3] float32."""
        K = np.asarray(self._cam.intrinsics, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"Expected intrinsics shape (3,3), got {K.shape}")
        return K

    def get_lidar(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return synchronized LiDAR point cloud [N,3] (XYZ camera coords).

        Returns None if no LiDAR is available for this frame.
        """
        if not (0 <= frame_idx < len(self._frames)):
            return None
        ts = self._frames[frame_idx][0]
        try:
            pts = self._scene.get_lidar_at_timestamp(ts)
            return np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        except Exception as exc:
            logger.warning(
                "get_lidar failed for frame_idx=%d (ts=%.3f): %s",
                frame_idx, ts, exc,
            )
            return None

    def get_gt_labels(self, frame_idx: int) -> list[GTLabel]:
        """Return ground truth labels for this frame."""
        if not (0 <= frame_idx < len(self._frames)):
            return []
        ts = self._frames[frame_idx][0]
        try:
            raw = self._scene.get_labels_at_timestamp(ts)
            return [
                GTLabel(
                    track_id=int(lb["track_id"]),
                    class_name=str(lb["class_name"]),
                    box_3d=[float(v) for v in lb["box_3d"]],
                )
                for lb in raw
            ]
        except Exception as exc:
            logger.warning(
                "get_gt_labels failed for frame_idx=%d (ts=%.3f): %s",
                frame_idx, ts, exc,
            )
            return []
