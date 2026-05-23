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
    ) -> list[Detection3D]:
        """
        Run detection on a single RGB frame [H,W,3] uint8.
        intrinsics: [3,3] float32, or None to estimate from frame size.
        Returns list of Detection3D sorted by score descending.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Move model off GPU and free memory."""
        ...
