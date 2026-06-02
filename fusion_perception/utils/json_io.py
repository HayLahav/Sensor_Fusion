"""Typed save/load for all pipeline intermediate JSON outputs."""
import json
import numpy as np
from pathlib import Path
from fusion_perception.utils.dataclasses import (
    Detection3D, Track, OccupancyGrid, SceneMemory, ReasoningOutput,
)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def _read(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_detections(
    path: Path,
    run_id: str,
    video_path: str,
    prompts: list[str],
    frames_data: dict[int, list[Detection3D]],
) -> None:
    payload = {
        "run_id": run_id,
        "video_path": str(video_path),
        "prompts": prompts,
        "frames": {
            str(idx): [d.to_dict() for d in dets]
            for idx, dets in frames_data.items()
        },
    }
    _write(path, payload)


def load_detections(path: Path) -> dict:
    return _read(path)


def save_tracks(
    path: Path,
    run_id: str,
    tracks: dict[int, Track],
) -> None:
    payload = {
        "run_id": run_id,
        "tracks": {str(tid): t.to_dict() for tid, t in tracks.items()},
    }
    _write(path, payload)


def load_tracks(path: Path) -> dict:
    return _read(path)


def save_occupancy(
    path: Path,
    run_id: str,
    config: dict,
    frames_data: dict[int, OccupancyGrid],
) -> None:
    payload = {
        "run_id": run_id,
        "config": config,
        "frames": {
            str(idx): {
                "frame_idx": g.frame_idx,
                "grid": g.grid,
                "occupied_cells": sum(
                    1 for row in g.grid for c in row if c > 0.5
                ),
                "free_cells": sum(
                    1 for row in g.grid for c in row if c <= 0.5
                ),
            }
            for idx, g in frames_data.items()
        },
    }
    _write(path, payload)


def save_reasoning(
    path: Path,
    run_id: str,
    reasoning_interval: int,
    outputs: list[ReasoningOutput],
) -> None:
    payload = {
        "run_id": run_id,
        "reasoning_interval": reasoning_interval,
        "outputs": [r.to_dict() for r in outputs],
    }
    _write(path, payload)


def load_reasoning(path: Path) -> dict:
    return _read(path)
