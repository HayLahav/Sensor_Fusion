"""Centroid-anchoring logic: match WildDet3D detections to CoWTracker query points.

Strategy: nearest-neighbour matching in 2D pixel space between detection
centroids and existing track query points. Hungarian assignment not used
here — NN is sufficient for the initial implementation.

TODO: Upgrade to Hungarian/IoU matching for crowded scenes.
"""
from __future__ import annotations
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D, Track


def _dist(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def match_detections_to_tracks(
    detections: list[Detection3D],
    active_tracks: dict[int, Track],
    nn_threshold: float,
) -> tuple[
    list[tuple[Detection3D, int]],   # matched: (detection, track_id)
    list[Detection3D],               # unmatched detections → new tracks
    list[int],                       # unmatched track IDs → occlusion
]:
    """
    Greedy nearest-neighbour matching between detection centroids and
    existing track query points.

    Processes detections in score order (highest first) to prioritise
    confident detections during assignment.
    """
    if not active_tracks:
        return [], list(detections), []

    matched: list[tuple[Detection3D, int]] = []
    unmatched_dets: list[Detection3D] = []
    used_track_ids: set[int] = set()

    for det in detections:
        best_tid = None
        best_dist = float("inf")
        for tid, track in active_tracks.items():
            if tid in used_track_ids:
                continue
            d = _dist(det.centroid_2d, track.cow_query_point)
            if d < best_dist:
                best_dist = d
                best_tid = tid

        if best_tid is not None and best_dist <= nn_threshold:
            matched.append((det, best_tid))
            used_track_ids.add(best_tid)
        else:
            unmatched_dets.append(det)

    unmatched_track_ids = [
        tid for tid in active_tracks if tid not in used_track_ids
    ]
    return matched, unmatched_dets, unmatched_track_ids


def assign_new_track_id(existing_tracks: dict) -> int:
    """Return the next available track ID (max existing + 1, or 1)."""
    if not existing_tracks:
        return 1
    return max(existing_tracks.keys()) + 1
