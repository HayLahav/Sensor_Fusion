"""Converts SceneMemory into structured Gemma prompts.

Uses txt templates from reasoning/templates/.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from fusion_perception.utils.dataclasses import SceneMemory, Track, OccupancyGrid
from fusion_perception.utils.geometry import world_to_grid

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_MAX_TRACKS_IN_PROMPT = 8


def _load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text()


# Cityscapes trainIds that unambiguously place a vehicle off the active road
_SEM_SURFACE: dict[int, str] = {
    1: "sidewalk",
    8: "off-road",   # vegetation
    9: "off-road",   # terrain
    2: "off-road",   # building
}
# Lateral distance (metres) beyond which a road-surface vehicle is on the shoulder
_SHOULDER_X_THRESHOLD = 3.5
_T_PRED           = 1.5   # seconds — prediction horizon shown in prompt
_SPEED_MIN        = 1.0   # m/s — below this a track is labelled stationary, no prediction
_MOTION_THRESHOLD = 0.3   # motion_grid value above which a cell is considered moving


def _surface_zone_tag(track: Track, grid: OccupancyGrid) -> str:
    """Combine BEV semantic class + lateral position into a single zone tag.

    Returns one of: lane | shoulder | sidewalk | off-road | edge | ""
    - lane     : road surface, within active lane width
    - shoulder : road surface but far from centre (parked on road side)
    - sidewalk : Cityscapes sidewalk class (clearly parked)
    - off-road : terrain / vegetation / building
    - edge     : sem unknown but lateral position suggests off-lane
    """
    if not track.position_3d_history or grid.sem_grid is None:
        return ""

    x_cam = track.position_3d_history[-1][0]
    z_cam = track.position_3d_history[-1][2]

    sem_class = -1
    rc = world_to_grid(x_cam, z_cam, grid.x_range, grid.z_range, grid.resolution)
    if rc is not None:
        row, col = rc
        try:
            sem_class = int(grid.sem_grid[row][col])
        except (IndexError, TypeError):
            pass

    # Unambiguous static surface → ignore lateral position
    if sem_class in _SEM_SURFACE:
        return f"[{_SEM_SURFACE[sem_class]}]"

    # Road surface: lateral position decides lane vs shoulder
    abs_x = abs(x_cam)
    if sem_class == 0:
        return "[shoulder]" if abs_x > _SHOULDER_X_THRESHOLD else "[lane]"

    # Unknown sem: still use lateral position as a weaker signal
    return "[edge]" if abs_x > _SHOULDER_X_THRESHOLD else "[lane]"


def _motion_label(track: Track, grid: OccupancyGrid | None = None) -> str:
    """Classify track motion using the BEV motion grid (primary) or KF velocity (fallback).

    The motion grid is computed as |current_occ - ego-warped_prev_occ|.  Because
    the warp uses ground-truth KITTI-360 poses, static objects cancel perfectly
    regardless of ego speed or turning angle.  Only truly moving objects produce
    a high residual.  KF velocity is used as a fallback when no motion grid is
    available, and for direction labelling when the object is confirmed moving.
    """
    # ── Primary: motion grid at track's BEV cell ─────────────────────────────
    motion_val = 0.0
    if grid is not None and grid.motion_grid is not None and track.position_3d_history:
        x_cam = track.position_3d_history[-1][0]
        z_cam = track.position_3d_history[-1][2]
        rc = world_to_grid(x_cam, z_cam, grid.x_range, grid.z_range, grid.resolution)
        if rc is not None:
            row, col = rc
            try:
                motion_val = float(grid.motion_grid[row][col])
            except (IndexError, TypeError):
                motion_val = 0.0

        if motion_val < _MOTION_THRESHOLD:
            return "stationary"

        # Confirmed moving — use velocity_3d for direction
        vx, _, vz = track.velocity_3d
        speed = (vx ** 2 + vz ** 2) ** 0.5
        if speed < 0.5:
            return "moving"
        if vz < -1.0:
            return "approaching"
        if vz > 1.0:
            return "receding"
        return "moving"

    # ── Fallback: KF velocity (no motion grid available) ─────────────────────
    vx, _, vz = track.velocity_3d
    speed = (vx ** 2 + vz ** 2) ** 0.5
    if speed < 1.0:
        return "stationary"
    if vz > 2.0:
        return "receding"
    if vz < -2.0:
        return "approaching"
    return "moving"


def _depth_band(depth: float) -> str:
    if depth < 15.0:
        return "near"
    if depth < 35.0:
        return "mid"
    return "far"


def _describe_track(
    track: Track,
    fps: float = 30.0,
    grid: OccupancyGrid | None = None,
) -> str:
    depth = track.position_3d_history[-1][2] if track.position_3d_history else 0.0
    tag   = _surface_zone_tag(track, grid) if grid is not None else ""
    motion = _motion_label(track, grid)

    vx, _, vz = track.velocity_3d
    speed = (vx ** 2 + vz ** 2) ** 0.5
    moving = speed >= _SPEED_MIN

    # Speed suffix — only for moving objects so static cars stay clean
    speed_str = f" {speed:.0f}m/s" if moving else ""

    # Predicted depth — forward/backward only (lateral drift not actionable at prompt level)
    pred_str = ""
    if moving and track.position_3d_history:
        pred_z = depth + vz * _T_PRED
        pred_z = max(0.0, min(80.0, pred_z))
        pred_str = f" →{pred_z:.0f}m@{_T_PRED:.1f}s"

    base = (
        f"#{track.track_id} {track.class_name} "
        f"{_depth_band(depth)} ({depth:.0f}m) {motion}{speed_str}"
    )
    return f"{base} {tag}{pred_str}" if tag else f"{base}{pred_str}"


def _bev_density(occupancy_pct: float) -> str:
    if occupancy_pct < 10.0:
        return "sparse"
    if occupancy_pct < 30.0:
        return "moderate"
    return "dense"


def build_scene_prompt(
    memory: SceneMemory,
    fps: float = 30.0,
    memory_prefix: str = "",
) -> str:
    """Build the primary scene summary prompt from SceneMemory."""
    template = _load_template("scene_summary.txt")

    tracks = memory.active_tracks[:_MAX_TRACKS_IN_PROMPT]
    grid = memory.occupancy_grid
    track_descriptions = "\n".join(
        _describe_track(t, fps, grid) for t in tracks
    ) or "No active tracks."
    if len(memory.active_tracks) > _MAX_TRACKS_IN_PROMPT:
        track_descriptions += f"\n(+{len(memory.active_tracks) - _MAX_TRACKS_IN_PROMPT} more)"

    total_cells = sum(len(row) for row in grid.grid)
    occupied_cells = sum(1 for row in grid.grid for c in row if c > 0.5)
    occupancy_pct = 100 * occupied_cells / max(total_cells, 1)

    event_str = "\n".join(memory.event_flags) if memory.event_flags else "None"
    surface_str = memory.sem_summary if memory.sem_summary else _bev_density(occupancy_pct)

    scene = template.format(
        frame_idx=memory.frame_idx,
        elapsed=memory.elapsed_seconds,
        n_tracks=len(memory.active_tracks),
        track_descriptions=track_descriptions,
        bev_density=_bev_density(occupancy_pct),
        surface=surface_str,
        event_flags=event_str,
    )
    if memory_prefix:
        return memory_prefix + scene
    return scene
