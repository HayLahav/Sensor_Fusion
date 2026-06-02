"""Abstract base class for 3D object detectors.

Implement this interface to swap WildDet3D for any future detector
without changing any downstream pipeline code.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D


class BaseDetector(ABC):

    @abstractmethod
    def load(self, checkpoint_path: str, device: str) -> None:
        """Load model weights onto device."""
        ...

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intrinsics: np.ndarray | None,
        prompts: list[str],
        lidar_pts_velo: np.ndarray | None = None,
        calib: object = None,
        sem_mask: np.ndarray | None = None,
        depth_map: np.ndarray | None = None,
    ) -> list[Detection3D]:
        """
        Run detection for one frame.
        frame          : [H,W,3] uint8 RGB
        intrinsics     : [3,3] float32, or None to estimate from frame size
        lidar_pts_velo : [N,4] float32 in Velodyne frame (x,y,z,intensity)
        calib          : KittiRawCalib or Kitti360Calib for coordinate transform
        sem_mask       : [H,W] int16 Cityscapes trainId map (from road segmentor)
        depth_map      : [H,W] float32 metric depth in metres (from depth estimator)
        Returns list of Detection3D sorted by score descending.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Move model off GPU and free memory."""
        ...
