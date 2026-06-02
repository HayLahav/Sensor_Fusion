"""Rolling cross-frame memory for Gemma scene reasoning.

Stores a compact window of past reasoning outputs and track lifecycle
events, then formats them as a prefix injected into each Gemma prompt
so the model understands scene continuity across frames.

Design constraints:
  - Prefix must be short (≤80 tokens) — Gemma max_new_tokens=100.
  - Data stored per entry: summary, first anomaly, trajectory, track IDs.
  - Track appeared/disappeared events are aggregated across the window.
  - Anomaly streak: how many consecutive reasoning calls flagged an anomaly.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from fusion_perception.utils.dataclasses import ReasoningOutput, Track

_DEFAULT_MAXLEN = 5   # how many past reasoning outputs to retain


@dataclass
class _MemoryEntry:
    frame_idx: int
    elapsed: float           # seconds
    summary: str
    anomaly: str             # first anomaly string, or ""
    trajectory: str
    n_tracks: int
    track_ids: frozenset     # set[int] at this moment


class GemmaMemoryBuffer:
    """Accumulates past reasoning outputs + track events for prompt injection.

    Usage (inside StageRunner):
        buf = GemmaMemoryBuffer()
        ...
        prefix = buf.format_prefix(frame_idx, fps)
        reasoning = gemma.reason(memory, vis, memory_prefix=prefix, ...)
        buf.add(reasoning, tracks, frame_idx, fps)
    """

    def __init__(self, maxlen: int = _DEFAULT_MAXLEN) -> None:
        self._entries: deque[_MemoryEntry] = deque(maxlen=maxlen)
        self._appeared: list[tuple[int, int, str]] = []   # (frame_idx, track_id, class_name)
        self._disappeared: list[tuple[int, int, str]] = []
        self._anomaly_streak: int = 0
        self._prev_track_ids: frozenset = frozenset()

    # ------------------------------------------------------------------
    def add(
        self,
        reasoning: "ReasoningOutput",
        tracks: "list[Track]",
        frame_idx: int,
        fps: float,
    ) -> None:
        """Record one reasoning output and update track lifecycle log."""
        current_ids: frozenset = frozenset(t.track_id for t in tracks)
        appeared = current_ids - self._prev_track_ids
        disappeared = self._prev_track_ids - current_ids

        id_to_cls = {t.track_id: t.class_name for t in tracks}
        for tid in appeared:
            cls = id_to_cls.get(tid, "object")
            self._appeared.append((frame_idx, tid, cls))
        for tid in disappeared:
            # class_name no longer in current tracks; mark unknown if needed
            self._disappeared.append((frame_idx, tid, "?"))
        self._prev_track_ids = current_ids

        # Trim event log — keep only events within the memory window
        if self._entries:
            oldest_frame = self._entries[0].frame_idx
        else:
            oldest_frame = frame_idx
        self._appeared = [e for e in self._appeared if e[0] >= oldest_frame]
        self._disappeared = [e for e in self._disappeared if e[0] >= oldest_frame]

        first_anomaly = reasoning.anomalies[0] if reasoning.anomalies else ""
        has_anomaly = bool(first_anomaly)
        self._anomaly_streak = (self._anomaly_streak + 1) if has_anomaly else 0

        self._entries.append(_MemoryEntry(
            frame_idx=frame_idx,
            elapsed=frame_idx / max(fps, 1.0),
            summary=reasoning.summary,
            anomaly=first_anomaly,
            trajectory=reasoning.trajectory_nl,
            n_tracks=len(tracks),
            track_ids=current_ids,
        ))

    # ------------------------------------------------------------------
    def format_prefix(self, current_frame_idx: int, fps: float) -> str:
        """Return a compact memory prefix string for the Gemma prompt.

        Returns "" if no past entries exist (first reasoning call).
        """
        if not self._entries:
            return ""

        lines: list[str] = ["=== Past observations ==="]

        for e in self._entries:
            t_sec = e.elapsed
            parts = [f"t={t_sec:.1f}s({e.n_tracks}obj): {e.summary}"]
            if e.anomaly:
                parts.append(f"CAUTION:{e.anomaly}")
            if e.trajectory:
                parts.append(f"trend:{e.trajectory}")
            lines.append(" | ".join(parts))

        # Track lifecycle events (last few only to stay compact)
        events: list[str] = []
        for fid, tid, cls in self._appeared[-3:]:
            t = fid / max(fps, 1.0)
            events.append(f"#{tid}({cls}) appeared@{t:.1f}s")
        for fid, tid, _ in self._disappeared[-3:]:
            t = fid / max(fps, 1.0)
            events.append(f"#{tid} left@{t:.1f}s")
        if events:
            lines.append("Events: " + ", ".join(events))

        # Cap streak display at 3 — higher counts add no new information and
        # cause Gemma to lock into "extreme caution" regardless of scene state.
        if 2 <= self._anomaly_streak <= 3:
            lines.append(f"Ongoing caution: {self._anomaly_streak} consecutive frames")

        lines.append("=========================")
        return "\n".join(lines) + "\n"
