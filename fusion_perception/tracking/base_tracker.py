"""Abstract base class for multi-object trackers."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D, Track


class BaseTracker(ABC):

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection3D],
        frame_idx: int,
    ) -> list[Track]:
        """Update tracker with new frame and detections. Return active tracks."""
        ...

    @abstractmethod
    def get_all_tracks(self) -> dict[int, Track]:
        """Return full track registry including inactive tracks."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all state between videos."""
        ...
