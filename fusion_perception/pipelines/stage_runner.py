"""Per-frame orchestration: calls each pipeline stage in order.

Detection pipeline:
  1. WildDet3D / CenterPoint → 3D boxes (metric depth, dimensions, orientation)
  2. fuse_lidar_with_sem → semantic class labels from road segmentation mask

Depth pipeline (when depth_estimator is supplied):
  3. DA-V2 monocular depth map → LiDAR global scale correction
  4. LidarDepthAnchor → per-detection depth replacement from LiDAR frustum
     or monocular depth patch (whichever has sufficient support)
  5. PointCloudAccumulator (optional) → dense world point cloud

Fallback (depth_estimator=None): WildDet3D depth, no correction.
"""
from __future__ import annotations
import numpy as np
from typing import Optional, TYPE_CHECKING
from fusion_perception.utils.dataclasses import (
    Detection3D, Track, OccupancyGrid, SceneMemory, ReasoningOutput
)
from fusion_perception.models.base_detector import BaseDetector
from fusion_perception.tracking.base_tracker import BaseTracker
from fusion_perception.occupancy.bev_grid import OccupancyBEVGenerator
from fusion_perception.tracking.trajectory_manager import SceneMemoryManager
from fusion_perception.models.gemma_wrapper import GemmaReasoningWrapper
from fusion_perception.visualization.frame_annotator import annotate_frame
from fusion_perception.visualization.bev_renderer import render_bev
from fusion_perception.visualization.output_compositor import composite
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.memory.gemma_memory import GemmaMemoryBuffer

if TYPE_CHECKING:
    from fusion_perception.segmentation.road_segmentor import RoadSegmentor
    from fusion_perception.depth.mono_depth_estimator import MonoDepthEstimator
    from fusion_perception.depth.quality_gate import DepthQualityGate
    from fusion_perception.depth.lidar_anchor import LidarDepthAnchor
    from fusion_perception.depth.reconstruction import PointCloudAccumulator

logger = get_logger("stage_runner")


class StageRunner:
    """Runs all pipeline stages for a single frame."""

    def __init__(
        self,
        detector: BaseDetector,
        tracker: BaseTracker,
        bev_generator: OccupancyBEVGenerator,
        scene_memory: SceneMemoryManager,
        gemma: GemmaReasoningWrapper,
        prompts: list[str],
        reasoning_interval: int,
        fps: float,
        visual_reasoning: bool,
        road_segmentor: "Optional[RoadSegmentor]" = None,
        road_seg_interval: int = 5,
        calib: object = None,
        depth_estimator: "Optional[MonoDepthEstimator]" = None,
        depth_quality_gate: "Optional[DepthQualityGate]" = None,
        lidar_anchor: "Optional[LidarDepthAnchor]" = None,
        point_cloud_acc: "Optional[PointCloudAccumulator]" = None,
    ) -> None:
        self.detector = detector                      # WildDet3D / CenterPoint (3D geometry)
        self.tracker = tracker
        self.bev_generator = bev_generator
        self.scene_memory = scene_memory
        self.gemma = gemma
        self.prompts = prompts
        self.reasoning_interval = reasoning_interval
        self.fps = fps
        self.visual_reasoning = visual_reasoning
        self.road_segmentor = road_segmentor
        self.road_seg_interval = road_seg_interval
        self.calib = calib
        self.depth_estimator = depth_estimator        # DA-V2 monocular depth
        self.depth_quality_gate = depth_quality_gate  # frame quality filter
        self.lidar_anchor = lidar_anchor              # LiDAR depth correction
        self.point_cloud_acc = point_cloud_acc        # rolling world point cloud
        self._last_reasoning: Optional[ReasoningOutput] = None
        self._sem_mask: Optional[np.ndarray] = None
        self._gemma_memory = GemmaMemoryBuffer(maxlen=5)

    def run_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intrinsics: Optional[np.ndarray] = None,
        lidar_pts: Optional[np.ndarray] = None,
        lidar_pts_velo: Optional[np.ndarray] = None,
        T_ego: Optional[np.ndarray] = None,
    ) -> dict:
        """Process one frame through all pipeline stages.

        Returns dict with keys:
            frame_idx, detections, tracks, occupancy, memory,
            reasoning, composite, static_events
        """
        # ── Semantic segmentation (road / BEV colour) ────────────────────────
        if self.road_segmentor is not None and frame_idx % self.road_seg_interval == 0:
            self._sem_mask = self.road_segmentor.segment(frame)

        # ── Monocular depth map (frame-only — no detection dependency) ────────
        # Runs before detection so detect() receives real depth for:
        #   - Method B (depth+sem fallback for 3D box fitting)
        #   - Keypoint yaw back-projection (car_keypoint.py)
        depth_map: Optional[np.ndarray] = None
        if self.depth_estimator is not None:
            run_depth = True
            if self.depth_quality_gate is not None:
                run_depth = self.depth_quality_gate.is_usable(frame)
            if run_depth:
                depth_map = self.depth_estimator.estimate(frame)
                if depth_map is not None and self.lidar_anchor is not None and intrinsics is not None and lidar_pts is not None:
                    depth_map = self.lidar_anchor.refine_depth_map(
                        depth_map, lidar_pts, intrinsics
                    )
                if depth_map is not None and self.point_cloud_acc is not None:
                    self.point_cloud_acc.update(depth_map, intrinsics)

        # ── Detection ────────────────────────────────────────────────────────
        from fusion_perception.models.detection_fusion import fuse_lidar_with_sem

        detections: list[Detection3D] = self.detector.detect(
            frame, frame_idx, intrinsics, self.prompts,
            lidar_pts_velo=lidar_pts_velo,
            calib=self.calib,
            sem_mask=self._sem_mask,
            depth_map=depth_map,
        )
        if self._sem_mask is not None and intrinsics is not None:
            detections = fuse_lidar_with_sem(
                detections, self._sem_mask, intrinsics, frame.shape[:2]
            )
        static_events: list[str] = []

        # ── LiDAR depth anchoring (refines detection depths post-detection) ──
        if depth_map is not None and self.lidar_anchor is not None and intrinsics is not None:
            detections = self.lidar_anchor.anchor_detections(
                detections, depth_map, lidar_pts, intrinsics
            )

        # ── Tracking ─────────────────────────────────────────────────────────
        tracks: list[Track] = self.tracker.update(
            frame, detections, frame_idx,
            fps=self.fps,
            intrinsics=intrinsics,
            T_ego=T_ego,
        )

        # ── BEV occupancy ────────────────────────────────────────────────────
        occupancy: OccupancyGrid = self.bev_generator.update(
            tracks, frame_idx,
            lidar_pts=lidar_pts,
            sem_mask=self._sem_mask,
            intrinsics=intrinsics,
            T_ego=T_ego,
        )

        # ── Scene memory ─────────────────────────────────────────────────────
        memory: SceneMemory = self.scene_memory.update(
            tracks, occupancy, frame_idx, self.fps
        )
        if static_events:
            memory.event_flags.extend(static_events)

        # ── Gemma reasoning ──────────────────────────────────────────────────
        # Annotate the frame for Gemma's rolling visual buffer every reasoning
        # interval. The wrapper buffers it regardless of visual_mode so the
        # buffer is warm if visual_mode is toggled. visual_reasoning flag is
        # kept for backward compat but the wrapper's own visual_mode governs
        # whether images are actually sent to the model.
        _kp_anns    = getattr(self.detector, 'last_kp_annotations', None)
        _inst_masks = getattr(self.detector, 'last_instance_masks', None)

        reasoning: Optional[ReasoningOutput] = None
        if frame_idx % self.reasoning_interval == 0:
            vis = annotate_frame(
                frame, tracks, detections, intrinsics=intrinsics,
                kp_annotations=_kp_anns, instance_masks=_inst_masks,
                occupancy=occupancy,
            )
            mem_prefix = self._gemma_memory.format_prefix(frame_idx, self.fps)
            reasoning = self.gemma.reason(
                memory, vis, trigger_reason="interval", fps=self.fps,
                memory_prefix=mem_prefix,
            )
            self._last_reasoning = reasoning
            if reasoning is not None:
                self._gemma_memory.add(reasoning, tracks, frame_idx, self.fps)

        # ── Visualisation ────────────────────────────────────────────────────
        annotated_frame = annotate_frame(
            frame, tracks, detections, intrinsics=intrinsics,
            kp_annotations=_kp_anns, instance_masks=_inst_masks,
            occupancy=occupancy,
        )
        bev_img = render_bev(occupancy, tracks)
        composite_img = composite(
            annotated_frame, bev_img, self._last_reasoning,
            depth_map=depth_map,
            sem_mask=self._sem_mask,
            show_sem=(self.road_segmentor is not None),
            kp_annotations=getattr(self.detector, 'last_kp_annotations', None),
            instance_masks=getattr(self.detector, 'last_instance_masks', None),
        )

        return {
            "frame_idx": frame_idx,
            "detections": detections,
            "tracks": tracks,
            "occupancy": occupancy,
            "memory": memory,
            "reasoning": reasoning,
            "composite": composite_img,
            "static_events": static_events,
            "depth_map": depth_map,
        }
