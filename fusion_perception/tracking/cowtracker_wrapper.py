"""CoWTracker centroid-anchoring tracker.

Uses CoWTrackerWindowed to track the 2D centroid of each active object
across frames. WildDet3D detections are matched to existing tracks by
nearest-neighbour in pixel space (see centroid_anchor.py).

Flow per frame:
  1. Run CoWTracker on frame window → get updated positions for all queries
  2. Match new WildDet3D detections to updated track positions (NN)
  3. Spawn new tracks for unmatched detections
  4. Increment occlusion counter for unmatched tracks; kill if > tolerance

TODO: Tune window_size vs. memory tradeoff for longer videos.
TODO: Add re-ID via appearance features for severe occlusions.
"""
from __future__ import annotations
import numpy as np
import torch
from collections import deque
from typing import Optional
from fusion_perception.tracking.base_tracker import BaseTracker
from fusion_perception.tracking.centroid_anchor import (
    match_detections_to_tracks, assign_new_track_id
)
from fusion_perception.utils.dataclasses import Detection3D, Track
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("cowtracker_wrapper")


class CoWTrackerWrapper(BaseTracker):
    """
    Wraps CoWTrackerWindowed with centroid-anchoring for object tracking.

    CoWTracker tracks a set of 2D query points across a sliding window of
    frames. Here, each query point corresponds to a WildDet3D object centroid.
    """

    def __init__(
        self,
        window_size: int = 8,
        max_tracks: int = 50,
        occlusion_tolerance: int = 10,
        nn_threshold: float = 50.0,
        device: str = "cuda",
    ) -> None:
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.occlusion_tolerance = occlusion_tolerance
        self.nn_threshold = nn_threshold
        self.device = device

        self._model = None
        self._all_tracks: dict[int, Track] = {}
        self._active_ids: set[int] = set()
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)

    def load(self) -> None:
        """Download and load CoWTracker model weights."""
        logger.info("Loading CoWTrackerWindowed from HuggingFace Hub")
        try:
            from cotracker.predictor import CoTrackerPredictor
        except ImportError:
            raise ImportError(
                "CoWTracker not installed. "
                "Run: pip install git+https://github.com/facebookresearch/co-tracker"
            )
        self._model = CoTrackerPredictor(
            checkpoint=None,  # auto-download
            window_len=self.window_size,
        ).to(self.device)
        self._model.eval()
        log_gpu_memory("CoWTracker loaded")
        logger.info("CoWTracker ready")

    @torch.no_grad()
    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection3D],
        frame_idx: int,
        **kwargs,
    ) -> list[Track]:
        """
        Update tracker state with a new frame and its detections.
        Returns list of currently active tracks.
        """
        self._frame_buffer.append(frame)

        active_tracks = {
            tid: self._all_tracks[tid]
            for tid in self._active_ids
            if tid in self._all_tracks
        }

        # Step 1: run CoWTracker to update existing query point positions
        if self._model is not None and len(active_tracks) > 0:
            active_tracks = self._run_cowtracker(active_tracks, frame_idx)

        # Step 2: match new detections to updated track positions
        matched, unmatched_dets, unmatched_tids = match_detections_to_tracks(
            detections=detections,
            active_tracks=active_tracks,
            nn_threshold=self.nn_threshold,
        )

        # Step 3: update matched tracks
        for det, tid in matched:
            track = self._all_tracks[tid]
            track.last_seen = frame_idx
            track.centroid_history.append(det.centroid_2d)
            track.position_3d_history.append(det.centroid_3d)
            track.cow_query_point = det.centroid_2d
            track.occlusion_count = 0
            track.is_active = True

        # Step 4: handle unmatched tracks (occlusion)
        for tid in unmatched_tids:
            track = self._all_tracks[tid]
            track.occlusion_count += 1
            if track.occlusion_count > self.occlusion_tolerance:
                track.is_active = False
                self._active_ids.discard(tid)
                logger.debug(f"Track {tid} killed after {self.occlusion_tolerance} occlusion frames")

        # Step 5: spawn new tracks for unmatched detections
        for det in unmatched_dets:
            if len(self._active_ids) >= self.max_tracks:
                logger.warning(f"max_tracks={self.max_tracks} reached, skipping new track")
                break
            new_id = assign_new_track_id(self._all_tracks)
            new_track = Track(
                track_id=new_id,
                class_name=det.class_name,
                first_seen=frame_idx,
                last_seen=frame_idx,
                centroid_history=[det.centroid_2d],
                position_3d_history=[det.centroid_3d],
                cow_query_point=det.centroid_2d,
                is_active=True,
                occlusion_count=0,
            )
            self._all_tracks[new_id] = new_track
            self._active_ids.add(new_id)
            logger.debug(f"New track {new_id}: {det.class_name} @ {det.centroid_2d}")

        active_list = [
            self._all_tracks[tid]
            for tid in self._active_ids
            if tid in self._all_tracks
        ]
        logger.debug(f"Frame {frame_idx}: {len(active_list)} active tracks")
        return active_list

    def _run_cowtracker(
        self,
        active_tracks: dict[int, Track],
        frame_idx: int,
    ) -> dict[int, Track]:
        """
        Run CoWTracker on the current frame buffer and update query points.
        Returns updated active_tracks dict with new cow_query_point values.
        """
        if len(self._frame_buffer) < 2:
            return active_tracks

        frames_np = np.stack(list(self._frame_buffer), axis=0)  # [T,H,W,3]
        video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        video = video.unsqueeze(0).to(self.device)  # [1,T,3,H,W]

        track_ids = list(active_tracks.keys())
        queries = torch.tensor(
            [[0.0, t.cow_query_point[0], t.cow_query_point[1]]
             for t in active_tracks.values()],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, N, 3]

        pred_tracks, pred_vis = self._model(video, queries=queries)
        # pred_tracks: [1, T, N, 2]

        last_pos = pred_tracks[0, -1]  # [N, 2] — positions in latest frame
        for i, tid in enumerate(track_ids):
            xy = last_pos[i].cpu().tolist()
            active_tracks[tid].cow_query_point = xy

        return active_tracks

    def get_all_tracks(self) -> dict[int, Track]:
        return self._all_tracks

    def reset(self) -> None:
        self._all_tracks = {}
        self._active_ids = set()
        self._frame_buffer.clear()
        logger.info("CoWTrackerWrapper reset")
