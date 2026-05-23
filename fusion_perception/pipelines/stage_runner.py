"""Per-frame orchestration: calls each pipeline stage in order."""
from __future__ import annotations
import numpy as np
from typing import Optional
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
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.bev_generator = bev_generator
        self.scene_memory = scene_memory
        self.gemma = gemma
        self.prompts = prompts
        self.reasoning_interval = reasoning_interval
        self.fps = fps
        self.visual_reasoning = visual_reasoning
        self._last_reasoning: Optional[ReasoningOutput] = None

    def run_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intrinsics: Optional[np.ndarray] = None,
        lidar_pts: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Process one frame through all stages.
        Returns dict with all stage outputs and composite visualization.
        """
        detections: list[Detection3D] = self.detector.detect(
            frame, frame_idx, intrinsics, self.prompts
        )
        tracks: list[Track] = self.tracker.update(
            frame, detections, frame_idx,
            fps=self.fps,
            intrinsics=intrinsics,
        )
        occupancy: OccupancyGrid = self.bev_generator.update(tracks, frame_idx, lidar_pts=lidar_pts)
        memory: SceneMemory = self.scene_memory.update(
            tracks, occupancy, frame_idx, self.fps
        )

        reasoning: Optional[ReasoningOutput] = None
        if frame_idx % self.reasoning_interval == 0:
            annotated = annotate_frame(frame, tracks, detections) if self.visual_reasoning else None
            reasoning = self.gemma.reason(
                memory, annotated, trigger_reason="interval", fps=self.fps
            )
            self._last_reasoning = reasoning

        annotated_frame = annotate_frame(
            frame, tracks, detections,
            reasoning_text=self._last_reasoning.summary if self._last_reasoning else "",
        )
        bev_img = render_bev(occupancy, tracks)
        composite_img = composite(annotated_frame, bev_img, self._last_reasoning)

        return {
            "frame_idx": frame_idx,
            "detections": detections,
            "tracks": tracks,
            "occupancy": occupancy,
            "memory": memory,
            "reasoning": reasoning,
            "composite": composite_img,
        }
