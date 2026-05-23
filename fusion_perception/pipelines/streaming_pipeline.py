"""Main streaming pipeline: initializes all models, runs frame loop.

All models stay resident in GPU for the full video duration.
JSON outputs are flushed every flush_interval frames.

TODO: Add SIGINT handler for graceful shutdown mid-video.
TODO: Support webcam/RTSP stream sources (VideoLoader already handles URL strings).
"""
from __future__ import annotations
import datetime
import cv2
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from fusion_perception.models.wilddet3d_wrapper import WildDet3DWrapper
from fusion_perception.tracking.cowtracker_wrapper import CoWTrackerWrapper
from fusion_perception.tracking.kalman_cowtracker import KalmanCoWTracker
from fusion_perception.occupancy.bev_grid import OccupancyBEVGenerator
from fusion_perception.tracking.trajectory_manager import SceneMemoryManager
from fusion_perception.models.gemma_wrapper import GemmaReasoningWrapper
from fusion_perception.pipelines.stage_runner import StageRunner
from fusion_perception.utils.video_loader import VideoLoader
from fusion_perception.utils.json_io import (
    save_detections, save_tracks, save_occupancy, save_reasoning
)
from fusion_perception.utils.memory_monitor import log_gpu_memory
from fusion_perception.utils.logging_setup import get_logger, setup_logging

logger = get_logger("streaming_pipeline")


class StreamingPipeline:
    """End-to-end streaming perception pipeline."""

    def __init__(self, config: DictConfig) -> None:
        self.cfg = config
        self.run_id = config.run_id or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        setup_logging(
            level=config.logging.level,
            log_file=config.logging.get("log_file"),
            use_rich=config.logging.use_rich,
        )
        self._output_base = Path(config.output.base_dir) / self.run_id
        self._output_base.mkdir(parents=True, exist_ok=True)

        self._all_detections: dict = {}
        self._all_tracks: dict = {}
        self._all_occupancy: dict = {}
        self._all_reasoning: list = []
        self._video_writer = None
        self._source_label: str = str(config.video.source)

    def _init_models(self) -> StageRunner:
        """Load all models onto GPU."""
        cfg = self.cfg
        logger.info("Initializing models...")

        detector = WildDet3DWrapper(
            score_threshold=cfg.detection.score_threshold,
            fp16=cfg.detection.fp16,
        )
        detector.load(cfg.detection.checkpoint, cfg.detection.device)

        backend = cfg.tracking.get("backend", "cowtracker")
        if backend == "kalman_cow":
            tracker = KalmanCoWTracker(
                window_size=cfg.tracking.window_size,
                max_tracks=cfg.tracking.max_tracks,
                lost_patience=cfg.tracking.get("lost_patience", 30),
                confirm_age=cfg.tracking.get("confirm_age", 3),
                high_score_threshold=cfg.tracking.get("high_score_threshold", 0.5),
                low_score_threshold=cfg.tracking.get("low_score_threshold", 0.2),
                assignment_cost_threshold=cfg.tracking.get("assignment_cost_threshold", 0.5),
                alpha_cost=cfg.tracking.get("alpha_cost", 0.35),
                cow_conf_threshold=cfg.tracking.get("cow_conf_threshold", 0.85),
                min_cow_points=cfg.tracking.get("min_cow_points", 4),
                velocity_decay=cfg.tracking.get("velocity_decay", 0.9),
                ego_motion=cfg.tracking.get("ego_motion", True),
                mahal_threshold=cfg.tracking.get("mahal_threshold", 9.21),
                lazy_cow_innovation=cfg.tracking.get("lazy_cow_innovation", 0.3),
                device=cfg.tracking.device,
            )
        else:
            tracker = CoWTrackerWrapper(
                window_size=cfg.tracking.window_size,
                max_tracks=cfg.tracking.max_tracks,
                occlusion_tolerance=cfg.tracking.get("occlusion_tolerance", 10),
                nn_threshold=cfg.tracking.get("nn_threshold", 50.0),
                device=cfg.tracking.device,
            )
        tracker.load()

        bev = OccupancyBEVGenerator(
            resolution=cfg.occupancy.resolution,
            x_range=list(cfg.occupancy.x_range),
            z_range=list(cfg.occupancy.z_range),
            decay_factor=cfg.occupancy.decay_factor,
            lidar_confidence=cfg.occupancy.lidar_confidence,
        )

        scene_memory = SceneMemoryManager()

        gemma = GemmaReasoningWrapper(
            model_id=cfg.reasoning.model_id,
            quantize_4bit=cfg.reasoning.quantize_4bit,
            max_new_tokens=cfg.reasoning.max_new_tokens,
            device=cfg.reasoning.device,
            visual_mode=cfg.reasoning.visual_mode,
        )
        if cfg.reasoning.enabled:
            gemma.load()

        log_gpu_memory("All models loaded")

        return StageRunner(
            detector=detector,
            tracker=tracker,
            bev_generator=bev,
            scene_memory=scene_memory,
            gemma=gemma,
            prompts=list(cfg.detection.prompts),
            reasoning_interval=cfg.reasoning.interval_frames,
            fps=30.0,
            visual_reasoning=cfg.reasoning.visual_mode,
        )

    def _run_frame_loop(
        self,
        runner,
        loader,
        intrinsics,
        lidar_fn,
        evaluator,
    ) -> None:
        """Shared per-frame loop used by both run() and run_py123d()."""
        flush_interval = self.cfg.output.flush_interval

        for frame_idx, frame, meta in loader:
            runner.fps = meta["fps"]

            lidar_pts = lidar_fn(frame_idx) if lidar_fn is not None else None
            outputs = runner.run_frame(
                frame, frame_idx,
                intrinsics=intrinsics,
                lidar_pts=lidar_pts,
            )

            self._all_detections[frame_idx] = outputs["detections"]
            for t in outputs["tracks"]:
                self._all_tracks[t.track_id] = t
            self._all_occupancy[frame_idx] = outputs["occupancy"]
            if outputs["reasoning"]:
                self._all_reasoning.append(outputs["reasoning"])

            if evaluator is not None:
                gt_labels = (
                    loader.get_gt_labels(frame_idx)
                    if hasattr(loader, "get_gt_labels") else []
                )
                evaluator.update(
                    frame_idx=frame_idx,
                    detections=outputs["detections"],
                    tracks=outputs["tracks"],
                    occupancy=outputs["occupancy"],
                    gt_labels=gt_labels,
                )

            if self._video_writer is not None:
                bgr = outputs["composite"][:, :, ::-1]
                self._video_writer.write(bgr)

            if (frame_idx + 1) % flush_interval == 0:
                self._flush_outputs()
                logger.info(f"Flushed outputs at frame {frame_idx}")

        self._flush_outputs()
        if self._video_writer:
            self._video_writer.release()

    def run(self, video_path: str) -> None:
        """Main frame loop for raw video files."""
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Processing video: {video_path}")

        runner = self._init_models()
        loader = VideoLoader(
            source=video_path,
            resize_hw=list(self.cfg.video.resize_hw),
            max_frames=self.cfg.video.max_frames,
        )

        if self.cfg.output.save_video:
            self._init_video_writer(loader)

        self._source_label = video_path
        self._run_frame_loop(
            runner=runner,
            loader=loader,
            intrinsics=None,
            lidar_fn=None,
            evaluator=None,
        )
        logger.info(f"Pipeline complete. Outputs in {self._output_base}")

    def run_py123d(
        self,
        log_dir: str,
        dataset_name: str,
        camera_name: str = "camera",
        evaluator=None,
    ) -> None:
        """Main frame loop for py123d-converted dataset logs."""
        from fusion_perception.data.py123d_loader import Py123dLoader

        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Processing py123d log: {log_dir} ({dataset_name})")

        runner = self._init_models()
        loader = Py123dLoader(log_dir=log_dir, camera_name=camera_name)
        intrinsics = loader.get_intrinsics()

        if self.cfg.output.save_video:
            logger.info("Video output not supported for py123d runs — skipping.")

        self._source_label = f"py123d://{log_dir}/{dataset_name}"
        self._run_frame_loop(
            runner=runner,
            loader=loader,
            intrinsics=intrinsics,
            lidar_fn=loader.get_lidar,
            evaluator=evaluator,
        )
        logger.info(f"Pipeline complete. Outputs in {self._output_base}")

    def _init_video_writer(self, loader: VideoLoader) -> None:
        fps = loader.fps or 30.0
        h, w = self.cfg.video.resize_hw
        out_path = str(self._output_base / "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = w + h + 200  # frame + BEV + panel
        self._video_writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, h))
        logger.info(f"Video writer: {out_path} ({out_w}x{h} @ {fps}fps)")

    def _flush_outputs(self) -> None:
        base = self._output_base
        cfg_dict = OmegaConf.to_container(self.cfg.occupancy, resolve=True)

        save_detections(
            base / "detections.json", self.run_id,
            getattr(self, "_source_label", str(self.cfg.video.source)), list(self.cfg.detection.prompts),
            self._all_detections,
        )
        save_tracks(base / "tracks.json", self.run_id, self._all_tracks)
        save_occupancy(base / "occupancy.json", self.run_id, cfg_dict, self._all_occupancy)
        save_reasoning(
            base / "reasoning.json", self.run_id,
            self.cfg.reasoning.interval_frames, self._all_reasoning,
        )
