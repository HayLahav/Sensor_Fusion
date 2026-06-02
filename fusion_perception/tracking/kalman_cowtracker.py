"""Hybrid Kalman Filter + CoWTracker for 3D multi-object tracking.

Per-frame flow:
  1. Predict all KF states forward (dt = 1/fps)
  2. Ego-motion compensation on track positions
  3. Run CoWTracker batch (lazy: skip stable tracks)
  4. Build hybrid cost matrix; Mahalanobis gate + class consistency gate
  5a. ByteTrack stage-1: confirmed+tentative tracks vs HIGH-score dets
  5b. ByteTrack stage-2: unmatched confirmed tracks vs LOW-score dets
  6. Handle misses for tracks not matched in either stage
  7. LOST tracks vs remaining low dets
  8. Spawn new tracks for unmatched high-conf detections only
  9. Prune LOST tracks exceeding lost_patience
"""
from __future__ import annotations
import numpy as np
import torch
from collections import deque
from typing import Optional

from fusion_perception.tracking.base_tracker import BaseTracker
from fusion_perception.tracking.track_state import TrackState, TrackStatus
from fusion_perception.tracking.kf_init import init_kf
from fusion_perception.tracking.measurement import synthesize_measurement, MeasurementConfig
from fusion_perception.tracking.cow_points import spawn_points, unpack_cow_outputs
from fusion_perception.tracking.ego_motion import estimate_homography  # noqa: F401 — kept for optional 2D fallback
from fusion_perception.tracking.association import (
    build_cost_matrix, mahalanobis_gate, hungarian_match
)
from fusion_perception.tracking.centroid_anchor import assign_new_track_id
from fusion_perception.utils.dataclasses import Detection3D, Track
from fusion_perception.utils.geometry import wrap_angle
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("kalman_cowtracker")


class KalmanCoWTracker(BaseTracker):
    """Hybrid 3D MOT: Kalman Filter with CoWTracker pixel-anchored measurement."""

    def __init__(
        self,
        window_size: int = 8,
        max_tracks: int = 50,
        lost_patience: int = 30,
        confirm_age: int = 3,
        high_score_threshold: float = 0.5,
        low_score_threshold: float = 0.2,
        assignment_cost_threshold: float = 0.65,
        alpha_cost: float = 0.35,
        cow_conf_threshold: float = 0.85,
        min_cow_points: int = 4,
        velocity_decay: float = 0.9,
        lazy_cow_innovation: float = 0.3,
        ego_motion: bool = True,
        mahal_threshold: float = 13.28,   # χ²(0.99, df=4) — gates on [cx,cy,cz,θ]
        device: str = "cuda",
    ) -> None:
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.lost_patience = lost_patience
        self.confirm_age = confirm_age
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.assignment_cost_threshold = assignment_cost_threshold
        self.alpha_cost = alpha_cost
        self.cow_conf_threshold = cow_conf_threshold
        self.min_cow_points = min_cow_points
        self.velocity_decay = velocity_decay
        self.lazy_cow_innovation = lazy_cow_innovation
        self.ego_motion = ego_motion
        self.mahal_threshold = mahal_threshold
        self.device = device

        self._cow_model = None
        self._tracks: dict[int, TrackState] = {}
        self._track_Q0: dict[int, np.ndarray] = {}  # base process noise per track
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._prev_frame: Optional[np.ndarray] = None
        self._meas_cfg = MeasurementConfig()
        self._fps: float = 10.0

    # ── BaseTracker interface ─────────────────────────────────────────────────

    @property
    def cow_active(self) -> bool:
        """True when CoWTracker loaded successfully; False = KF-only degraded mode."""
        return self._cow_model is not None

    def load(self) -> None:
        logger.info("Loading CoWTracker for KalmanCoWTracker")
        try:
            from cotracker.predictor import CoTrackerPredictor
            self._cow_model = CoTrackerPredictor(
                checkpoint=None, window_len=self.window_size
            ).to(self.device)
            self._cow_model.eval()
            log_gpu_memory("CoWTracker loaded (KalmanCoWTracker)")
            logger.info("CoWTracker ready — hybrid KF+pixel mode active")
        except ImportError as exc:
            self._cow_model = None
            logger.error(
                "CoWTracker import failed — tracker running in DEGRADED KF-only mode. "
                "Pixel anchoring is disabled; tracking quality will be reduced. "
                "Install with: pip install git+https://github.com/facebookresearch/co-tracker\n"
                f"Original error: {exc}"
            )
        except Exception as exc:
            self._cow_model = None
            logger.error(
                f"CoWTracker failed to load ({type(exc).__name__}: {exc}) — "
                "tracker running in DEGRADED KF-only mode."
            )

    @torch.no_grad()
    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection3D],
        frame_idx: int,
        fps: float = 10.0,
        intrinsics: Optional[np.ndarray] = None,
        **kwargs,
    ) -> list[Track]:
        self._fps = fps
        dt = 1.0 / max(fps, 1.0)
        self._frame_buffer.append(frame)

        K = intrinsics if intrinsics is not None else np.eye(3, dtype=np.float32)

        # 1. Predict all KF states
        self._predict_all(dt)

        # 2. Ego-motion compensation
        T_ego = kwargs.get('T_ego', None)
        if T_ego is not None:
            self._apply_ego_motion(T_ego, dt)
        self._prev_frame = frame.copy()

        # 3. Split detections by score (ByteTrack two-threshold)
        high_dets = [d for d in detections if d.score >= self.high_score_threshold]
        low_dets = [d for d in detections
                    if self.low_score_threshold <= d.score < self.high_score_threshold]

        # 4. CoWTracker batch update (lazy: skip tracks with small innovation)
        cow_displacements, cow_valids = self._run_cow_batch(frame_idx)

        # 5a. Stage-1: confirmed+tentative tracks vs HIGH dets (defer misses)
        confirmed_ids = [tid for tid, ts in self._tracks.items()
                         if ts.status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE)]
        matched_high, unmatched_confirmed = self._associate_and_update(
            confirmed_ids, high_dets, cow_displacements, cow_valids, K, frame_idx,
            handle_misses=False,
        )

        # 5b. Stage-2 (ByteTrack): unmatched confirmed tracks vs LOW dets
        matched_low, still_unmatched = self._associate_and_update(
            unmatched_confirmed, low_dets, cow_displacements, cow_valids, K, frame_idx,
            handle_misses=False,
        )

        # 6. Handle misses for all tracks not matched in either stage
        for tid in still_unmatched:
            self._handle_miss(tid)

        # 7. LOST tracks vs remaining unmatched low dets
        unmatched_low = [d for j, d in enumerate(low_dets) if j not in matched_low]
        lost_ids = [tid for tid, ts in self._tracks.items()
                    if ts.status == TrackStatus.LOST]
        self._associate_and_update(
            lost_ids, unmatched_low, cow_displacements, cow_valids, K, frame_idx,
            handle_misses=True,
        )

        # 8. Spawn new tracks for unmatched HIGH dets only
        for j, det in enumerate(high_dets):
            if j not in matched_high and len(self._tracks) < self.max_tracks:
                self._spawn_track(det, frame_idx, dt)

        # 9. Prune dead LOST tracks
        to_delete = [
            tid for tid, ts in self._tracks.items()
            if ts.status == TrackStatus.LOST and ts.miss_count > self.lost_patience
        ]
        for tid in to_delete:
            del self._tracks[tid]
            self._track_Q0.pop(tid, None)
            logger.debug(f"Deleted track {tid} after {self.lost_patience} lost frames")

        active = [
            ts.to_track() for ts in self._tracks.values()
            if ts.status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE)
        ]
        logger.debug(f"Frame {frame_idx}: {len(active)} active tracks")
        return active

    def get_all_tracks(self) -> dict:
        return self._tracks

    def reset(self) -> None:
        self._tracks = {}
        self._track_Q0 = {}
        self._frame_buffer.clear()
        self._prev_frame = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_ego_motion(self, T_ego: np.ndarray, dt: float) -> None:
        """Shift every track's KF position by the ego transform.

        Position is fully corrected (rotation + translation) so stationary
        world objects stay at their correct world-frame position.
        Velocity is left to the BEV motion-grid for motion classification —
        KF velocity is too noisy to reliably separate ego motion from object
        motion, especially during turns.
        """
        R = T_ego[:3, :3]
        t = T_ego[:3, 3]
        for ts in self._tracks.values():
            ts.kf.x[:3, 0]  = R @ ts.kf.x[:3, 0] + t
            ts.kf.x[7:10, 0] = R @ ts.kf.x[7:10, 0]

    def _predict_all(self, dt: float) -> None:
        for ts in self._tracks.values():
            ts.kf.F[0, 7] = dt
            ts.kf.F[1, 8] = dt
            ts.kf.F[2, 9] = dt
            ts.kf.predict()
            ts.age += 1
            if ts.status == TrackStatus.LOST:
                ts.kf.x[7:, 0] *= self.velocity_decay

    def _run_cow_batch(
        self, frame_idx: int
    ) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
        if self._cow_model is None or len(self._frame_buffer) < 2:
            return {}, {}

        active_ids = [
            tid for tid, ts in self._tracks.items()
            if ts.cow_points_abs is not None
            and (ts.innovation_norm > self.lazy_cow_innovation
                 or frame_idx % 3 == 0)
        ]
        if not active_ids:
            return {}, {}

        all_points = [self._tracks[tid].cow_points_abs for tid in active_ids]
        point_counts = [len(p) for p in all_points]

        query_unpadded = []
        for pts in all_points:
            for xy in pts:
                query_unpadded.append([0.0, float(xy[0]), float(xy[1])])

        frames_np = np.stack(list(self._frame_buffer), axis=0)
        video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        video = video.unsqueeze(0).to(self.device)

        queries = torch.tensor(query_unpadded, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

        pred_tracks, pred_vis = self._cow_model(video, queries=queries)

        return unpack_cow_outputs(
            pred_tracks, pred_vis, active_ids, point_counts,
            conf_threshold=self.cow_conf_threshold,
            min_points=self.min_cow_points,
        )

    def _associate_and_update(
        self,
        track_ids: list[int],
        dets: list[Detection3D],
        cow_displacements: dict[int, np.ndarray],
        cow_valids: dict[int, bool],
        K: np.ndarray,
        frame_idx: int,
        *,
        handle_misses: bool = True,
    ) -> tuple[set[int], list[int]]:
        """Associate track_ids with dets, update matched.

        Returns (matched_det_indices, unmatched_track_ids).
        When handle_misses=False the caller is responsible for calling
        _handle_miss on the returned unmatched_track_ids.
        """
        matched_det_indices: set[int] = set()

        if not track_ids:
            return matched_det_indices, []

        if not dets:
            if handle_misses:
                for tid in track_ids:
                    self._handle_miss(tid)
            return matched_det_indices, list(track_ids)

        pred_boxes = [self._tracks[tid].kf.x[:7, 0].tolist() for tid in track_ids]
        # Reorder Detection3D box_3d [cx,cy,cz,w,h,l,ry] → KF format [cx,cy,cz,θ,l,w,h]
        det_boxes = [
            [d.box_3d[0], d.box_3d[1], d.box_3d[2],
             d.box_3d[6], d.box_3d[5], d.box_3d[3], d.box_3d[4]]
            for d in dets
        ]

        pred_states = [self._tracks[tid].kf.x.flatten() for tid in track_ids]
        pred_covs = [self._tracks[tid].kf.P for tid in track_ids]
        det_z = [np.array(db, dtype=np.float64) for db in det_boxes]
        gate_mask = mahalanobis_gate(pred_states, pred_covs, det_z, self.mahal_threshold)

        cow_valid_set = {i for i, tid in enumerate(track_ids) if cow_valids.get(tid, False)}
        C = build_cost_matrix(pred_boxes, det_boxes, cow_valid_set, self.alpha_cost)
        C[~gate_mask] = 1e6

        # Class consistency gate: block cross-class assignments
        for i, tid in enumerate(track_ids):
            track_cls = self._tracks[tid].class_name
            for j, det in enumerate(dets):
                if track_cls != det.class_name:
                    C[i, j] = 1e6

        matched, unmatched_rows, _ = hungarian_match(C, threshold=self.assignment_cost_threshold)

        for row_i, col_j in matched:
            tid = track_ids[row_i]
            det = dets[col_j]
            matched_det_indices.add(col_j)
            self._update_track(tid, det, cow_displacements, cow_valids, K, frame_idx)

        unmatched_track_ids = [track_ids[r] for r in unmatched_rows]
        if handle_misses:
            for tid in unmatched_track_ids:
                self._handle_miss(tid)

        return matched_det_indices, unmatched_track_ids

    def _update_track(
        self,
        tid: int,
        det: Detection3D,
        cow_displacements: dict,
        cow_valids: dict,
        K: np.ndarray,
        frame_idx: int,
    ) -> None:
        ts = self._tracks[tid]
        cow_disp = cow_displacements.get(tid)
        cow_valid = cow_valids.get(tid, False)

        z, R = synthesize_measurement(det, cow_disp, cow_valid, K, self._meas_cfg)

        # Yaw ambiguity correction: avoid ±π flip
        pred_theta = float(ts.kf.x[3, 0])
        z[3] = wrap_angle(z[3])
        delta_theta = wrap_angle(z[3] - pred_theta)
        if abs(delta_theta) > np.pi / 2:
            z[3] = wrap_angle(z[3] + np.pi)

        ts.kf.R = R
        ts.kf.update(z)
        ts.kf.x[3, 0] = wrap_angle(ts.kf.x[3, 0])

        # Adaptive Q: inflate base Q by innovation norm, reset each step to avoid drift
        innovation = z - (ts.kf.H @ ts.kf.x).flatten()
        ts.innovation_norm = float(np.linalg.norm(innovation))
        q_factor = 1.0 + 0.1 * ts.innovation_norm
        Q0 = self._track_Q0.get(tid)
        if Q0 is not None:
            ts.kf.Q = Q0 * q_factor

        # Dimension EMA smoothing (l, w, h)
        alpha_dim = 0.15
        for kf_idx, det_val in zip([4, 5, 6], [det.box_3d[5], det.box_3d[3], det.box_3d[4]]):
            ts.kf.x[kf_idx, 0] = (1 - alpha_dim) * ts.kf.x[kf_idx, 0] + alpha_dim * det_val

        if ts.status == TrackStatus.CONFIRMED:
            ts.cow_points_abs = spawn_points(det.box_2d)
            ts.last_bbox2d = det.box_2d

        ts.last_box3d = ts.kf.x[:7, 0].copy()
        ts.last_seen = frame_idx
        ts.miss_count = 0
        ts.centroid_history.append(det.centroid_2d)
        ts.position_3d_history.append(det.centroid_3d)

        if ts.status == TrackStatus.TENTATIVE:
            ts.confirm_hits += 1
            if ts.confirm_hits >= self.confirm_age:
                ts.status = TrackStatus.CONFIRMED
        elif ts.status == TrackStatus.LOST:
            # Re-localization: inflate P to accept new visual measurement
            ts.kf.P *= 3.0
            ts.status = TrackStatus.CONFIRMED
            ts.miss_count = 0

    def _handle_miss(self, tid: int) -> None:
        if tid not in self._tracks:
            return
        ts = self._tracks[tid]
        ts.miss_count += 1
        if ts.status == TrackStatus.CONFIRMED and ts.miss_count > 2:
            ts.status = TrackStatus.LOST
            ts.cow_points_abs = None
            ts.cow_points_rel = None
        elif ts.status == TrackStatus.TENTATIVE and ts.miss_count > 1:
            del self._tracks[tid]
            self._track_Q0.pop(tid, None)

    def _spawn_track(self, det: Detection3D, frame_idx: int, dt: float) -> None:
        new_id = assign_new_track_id(self._tracks)
        # Reorder box_3d [cx,cy,cz,w,h,l,ry] → KF format [cx,cy,cz,θ,l,w,h]
        box3d = [
            det.box_3d[0], det.box_3d[1], det.box_3d[2],
            det.box_3d[6], det.box_3d[5], det.box_3d[3], det.box_3d[4]
        ]
        kf = init_kf(box3d, dt=dt)
        self._track_Q0[new_id] = kf.Q.copy()
        ts = TrackState(
            track_id=new_id,
            class_name=det.class_name,
            kf=kf,
            status=TrackStatus.TENTATIVE,
            first_seen=frame_idx,
            last_seen=frame_idx,
            last_box3d=np.array(box3d),
            cow_points_abs=spawn_points(det.box_2d),
            last_bbox2d=det.box_2d,
            centroid_history=[det.centroid_2d],
            position_3d_history=[det.centroid_3d],
        )
        self._tracks[new_id] = ts
        logger.debug(f"New track {new_id}: {det.class_name} @ z={det.centroid_3d[2]:.1f}m")
