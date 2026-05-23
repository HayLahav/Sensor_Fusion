"""Scene memory: accumulates track + occupancy state, detects events.

Events currently detected:
  - new_object:<track_id>      track just spawned
  - lost_object:<track_id>     track just killed
  - sudden_stop:<track_id>     object velocity near zero after moving

TODO: Add sudden_appearance (object enters from frame edge).
TODO: Add trajectory_crossing (two track paths intersect).
"""
from __future__ import annotations
import numpy as np
from fusion_perception.utils.dataclasses import Track, OccupancyGrid, SceneMemory
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("trajectory_manager")


class SceneMemoryManager:
    """Aggregates perception state across frames. Detects semantic events."""

    def __init__(self, sudden_stop_threshold: float = 2.0) -> None:
        self.sudden_stop_threshold = sudden_stop_threshold
        self._prev_active_ids: set[int] = set()
        self._event_flags: list[str] = []
        self._frame_count: int = 0
        self._snapshot: SceneMemory | None = None

    def update(
        self,
        tracks: list[Track],
        occupancy: OccupancyGrid,
        frame_idx: int,
        fps: float,
    ) -> SceneMemory:
        """Update state, detect events, return current SceneMemory."""
        self._frame_count += 1
        current_ids = {t.track_id for t in tracks if t.is_active}

        # Detect new objects
        for tid in current_ids - self._prev_active_ids:
            self._event_flags.append(f"new_object:{tid}")
            logger.info(f"Event: new_object:{tid}")

        # Detect lost objects
        for tid in self._prev_active_ids - current_ids:
            self._event_flags.append(f"lost_object:{tid}")
            logger.info(f"Event: lost_object:{tid}")

        # Detect sudden stops
        for track in tracks:
            if len(track.centroid_history) >= 3:
                recent = np.array(track.centroid_history[-3:])
                velocities = np.linalg.norm(np.diff(recent, axis=0), axis=1)
                if (velocities[-1] < self.sudden_stop_threshold
                        and velocities[0] > self.sudden_stop_threshold * 3):
                    flag = f"sudden_stop:{track.track_id}"
                    if flag not in self._event_flags[-10:]:
                        self._event_flags.append(flag)

        self._prev_active_ids = current_ids
        elapsed = frame_idx / max(fps, 1.0)

        self._snapshot = SceneMemory(
            frame_idx=frame_idx,
            active_tracks=tracks,
            occupancy_grid=occupancy,
            event_flags=list(self._event_flags[-20:]),  # keep last 20 events
            frame_count=self._frame_count,
            elapsed_seconds=elapsed,
        )
        return self._snapshot

    def get_snapshot(self) -> SceneMemory | None:
        return self._snapshot

    def reset(self) -> None:
        self._prev_active_ids = set()
        self._event_flags = []
        self._frame_count = 0
        self._snapshot = None
