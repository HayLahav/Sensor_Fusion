"""Shared dataclasses for the fusion perception pipeline.

All types are JSON-serializable via dataclasses-json.
These are the contracts between every pipeline stage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Optional


@dataclass_json
@dataclass
class Detection3D:
    """Single 3D detection from WildDet3D for one frame."""
    frame_idx: int
    class_id: int
    class_name: str
    score: float
    score_2d: float
    score_3d: float
    box_2d: list[float]          # [x1, y1, x2, y2] pixels
    box_3d: list[float]          # [cx, cy, cz, w, h, l, ry] camera coords
    centroid_2d: list[float]     # [x, y] pixels — center of box_2d
    centroid_3d: list[float]     # [x, y, z] camera coords
    depth: float                 # estimated depth in meters


@dataclass_json
@dataclass
class Track:
    """Persistent object identity across frames."""
    track_id: int
    class_name: str
    first_seen: int
    last_seen: int
    centroid_history: list[list[float]]        # [[x,y], ...] pixels per frame
    position_3d_history: list[list[float]]     # [[x,y,z], ...] camera coords
    cow_query_point: list[float]               # current CoWTracker query [x, y]
    is_active: bool
    occlusion_count: int                       # frames since last matched detection


@dataclass_json
@dataclass
class OccupancyGrid:
    """BEV occupancy grid state at a single frame."""
    frame_idx: int
    resolution: float               # meters per cell
    x_range: list[float]            # [x_min, x_max] meters
    z_range: list[float]            # [z_min, z_max] meters (forward)
    grid: list[list[float]]         # 2D array: 0.0=free, 1.0=occupied
    decay_factor: float


@dataclass_json
@dataclass
class SceneMemory:
    """Accumulated perception state across all frames up to now."""
    frame_idx: int
    active_tracks: list[Track]
    occupancy_grid: OccupancyGrid
    event_flags: list[str]          # e.g. ["sudden_stop:2", "new_object:5"]
    frame_count: int
    elapsed_seconds: float


@dataclass_json
@dataclass
class ReasoningOutput:
    """Gemma reasoning result for a given frame window."""
    frame_idx: int
    trigger_reason: str             # "interval" | "event"
    visual_mode: bool
    prompt_used: str
    summary: str
    anomalies: list[str]
    trajectory_nl: str
    raw_response: str
    latency_ms: float


@dataclass_json
@dataclass
class GTLabel:
    """Ground truth label from py123d dataset."""
    track_id: int
    class_name: str
    box_3d: list[float]   # [cx, cy, cz, w, h, l, ry] camera coords


@dataclass_json
@dataclass
class BenchmarkResult:
    """Aggregated metrics for one pipeline run on one dataset log."""
    dataset: str
    log_id: str
    map: float
    mota: float
    motp: float
    mean_occ_iou: float
    per_class_ap: dict[str, float]
    per_frame_occ_iou: list[float]
