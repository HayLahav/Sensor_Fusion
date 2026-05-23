"""Converts SceneMemory into structured Gemma prompts.

Uses txt templates from reasoning/templates/.
TODO: Add velocity estimation from centroid history deltas.
TODO: Support custom template injection via config.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from fusion_perception.utils.dataclasses import SceneMemory, Track

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text()


def _describe_track(track: Track, fps: float = 30.0) -> str:
    """Convert a Track to a single-line natural language description."""
    n_hist = len(track.centroid_history)
    if n_hist >= 2:
        delta = np.array(track.centroid_history[-1]) - np.array(track.centroid_history[-2])
        speed_px = float(np.linalg.norm(delta))
    else:
        speed_px = 0.0

    depth = track.position_3d_history[-1][2] if track.position_3d_history else 0.0
    return (
        f"Track {track.track_id} ({track.class_name}): "
        f"depth={depth:.1f}m, speed={speed_px:.1f}px/frame, "
        f"seen {track.last_seen - track.first_seen + 1} frames"
    )


def build_scene_prompt(memory: SceneMemory, fps: float = 30.0) -> str:
    """Build the primary scene summary prompt from SceneMemory."""
    template = _load_template("scene_summary.txt")

    track_descriptions = "\n".join(
        _describe_track(t, fps) for t in memory.active_tracks
    ) or "No active tracks."

    grid = memory.occupancy_grid
    total_cells = sum(len(row) for row in grid.grid)
    occupied_cells = sum(1 for row in grid.grid for c in row if c > 0.5)
    occupancy_pct = 100 * occupied_cells / max(total_cells, 1)

    event_str = "\n".join(memory.event_flags) if memory.event_flags else "None"

    return template.format(
        frame_idx=memory.frame_idx,
        elapsed=memory.elapsed_seconds,
        n_tracks=len(memory.active_tracks),
        track_descriptions=track_descriptions,
        occupied_cells=occupied_cells,
        total_cells=total_cells,
        occupancy_pct=occupancy_pct,
        event_flags=event_str,
    )
