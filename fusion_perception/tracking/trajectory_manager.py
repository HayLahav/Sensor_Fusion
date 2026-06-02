"""Scene memory: accumulates track + occupancy state, detects events.

Events reported per frame (compact — no stale accumulation):
  - new_object / new_objects:N    tracks just confirmed this frame
  - lost_object / lost_objects:N  confirmed tracks lost this frame
  - sudden_stop:<track_id>        object decelerated sharply (3D velocity)

New/lost churn is summarised as a count so the Gemma prompt stays clean.
Listing individual IDs for every spawn/loss floods the context with tracker
noise rather than meaningful driving events.
"""
from __future__ import annotations
import numpy as np
from fusion_perception.utils.dataclasses import Track, OccupancyGrid, SceneMemory
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("trajectory_manager")


class SceneMemoryManager:
    """Aggregates perception state across frames. Detects semantic events."""

    def __init__(self, min_confirm_frames: int = 3) -> None:
        self.min_confirm_frames = min_confirm_frames
        self._prev_active_ids: set[int] = set()
        self._confirmed_ids: set[int] = set()
        self._recent_stops: set[str] = set()   # cleared each frame
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
        frame_events: list[str] = []

        # ── New confirmed tracks this frame ───────────────────────────────────
        n_new = 0
        for t in tracks:
            if t.is_active and t.track_id not in self._confirmed_ids:
                age = frame_idx - t.first_seen
                if age >= self.min_confirm_frames:
                    self._confirmed_ids.add(t.track_id)
                    n_new += 1
                    logger.info(f"Event: new_object:{t.track_id} (age={age})")

        # ── Lost confirmed tracks this frame ──────────────────────────────────
        n_lost = 0
        for tid in self._prev_active_ids - current_ids:
            if tid in self._confirmed_ids:
                n_lost += 1
                logger.info(f"Event: lost_object:{tid}")

        # Summarise churn as counts — individual IDs clutter the prompt
        if n_new == 1:
            frame_events.append("new_object")
        elif n_new > 1:
            frame_events.append(f"new_objects:{n_new}")
        if n_lost == 1:
            frame_events.append("lost_object")
        elif n_lost > 1:
            frame_events.append(f"lost_objects:{n_lost}")

        # ── Sudden stops (3D position, metres/frame) ──────────────────────────
        self._recent_stops.clear()
        for track in tracks:
            if len(track.position_3d_history) >= 3:
                recent_3d = np.array(track.position_3d_history[-3:])
                velocities_3d = np.linalg.norm(np.diff(recent_3d, axis=0), axis=1)
                # was_moving: > 0.5 m/frame ≈ 5 m/s at 10 fps
                # now_stopped: < 0.1 m/frame ≈ 1 m/s at 10 fps
                if velocities_3d[-1] < 0.1 and velocities_3d[0] > 0.5:
                    flag = f"sudden_stop:{track.track_id}"
                    if flag not in self._recent_stops:
                        self._recent_stops.add(flag)
                        frame_events.append(flag)
                        logger.info(f"Event: {flag}")

        self._prev_active_ids = current_ids
        elapsed = frame_idx / max(fps, 1.0)

        self._snapshot = SceneMemory(
            frame_idx=frame_idx,
            active_tracks=tracks,
            occupancy_grid=occupancy,
            event_flags=frame_events,   # current frame only — no stale accumulation
            frame_count=self._frame_count,
            elapsed_seconds=elapsed,
            sem_summary=occupancy.sem_summary,
        )
        return self._snapshot

    def get_snapshot(self) -> SceneMemory | None:
        return self._snapshot

    def reset(self) -> None:
        self._prev_active_ids = set()
        self._confirmed_ids = set()
        self._recent_stops = set()
        self._frame_count = 0
        self._snapshot = None
