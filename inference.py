#!/usr/bin/env python
"""
inference.py — Fusion Perception Pipeline CLI

Two-pass approach (WildDet3D + CoWTracker) then (Gemma 4):
  Pass 1: detection + tracking + BEV occupancy over every frame
  Pass 2: Gemma reasoning on collected SceneMemory snapshots

Usage:
    python inference.py --video /path/to/video.mp4
    python inference.py --video clip.mp4 --kitti-root /data/kitti/seq --max-frames 200
    python inference.py --video clip.mp4 --kitti360-root /data/KITTI-360 \
                        --kitti360-seq 2013_05_28_drive_0000_sync --eval
"""
from __future__ import annotations
import argparse
import datetime
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fusion Perception Pipeline — WildDet3D + CoWTracker + Gemma 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", default=None,
                   help="Input video path (.mp4). Omit when using --kitti360-root + --kitti360-seq.")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Base YAML config file")
    p.add_argument("--colab-config", default=None,
                   help="Optional second YAML to merge (e.g. configs/colab.yaml)")
    p.add_argument("--output", default="outputs",
                   help="Output root directory")
    p.add_argument("--run-id", default=None,
                   help="Run identifier; auto-generated if omitted")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap number of frames (None = full video)")
    # LiDAR
    p.add_argument("--kitti-root", default=None,
                   help="KITTI sequence root (for LiDAR .bin loading)")
    p.add_argument("--kitti360-root", default=None,
                   help="KITTI-360 dataset root (for LiDAR + GT evaluation)")
    p.add_argument("--kitti360-seq", default=None,
                   help="KITTI-360 sequence name, e.g. 2013_05_28_drive_0000_sync")
    # Evaluation
    p.add_argument("--eval", action="store_true",
                   help="Run evaluation against KITTI-360 GT labels")
    p.add_argument("--eval-every", type=int, default=5,
                   help="Evaluate every N frames (GT label loading is slow)")
    # Debug output
    p.add_argument("--save-debug-frames", action="store_true",
                   help="Save annotated debug frames every --debug-every frames")
    p.add_argument("--debug-every", type=int, default=50,
                   help="Save a debug frame image every N processed frames")
    p.add_argument("--no-video", action="store_true",
                   help="Skip composite video output (faster)")
    # Intrinsics override
    p.add_argument("--fx", type=float, default=None, help="Camera focal length fx")
    p.add_argument("--fy", type=float, default=None, help="Camera focal length fy")
    p.add_argument("--cx-px", type=float, default=None, help="Principal point cx")
    p.add_argument("--cy-px", type=float, default=None, help="Principal point cy")
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(base: str, extra: str | None):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(base)
    if extra:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(extra))
    return cfg


def _build_intrinsics(args, frame_h: int, frame_w: int):
    import numpy as np
    from fusion_perception.utils.geometry import estimate_intrinsics
    if args.fx is not None:
        cx = args.cx_px if args.cx_px is not None else frame_w / 2.0
        cy = args.cy_px if args.cy_px is not None else frame_h / 2.0
        fy = args.fy if args.fy is not None else args.fx
        return np.array([[args.fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return estimate_intrinsics(frame_h, frame_w)


def _save_debug_frame(outputs: dict, frame_idx: int, debug_dir: Path) -> None:
    import cv2
    img_bgr = outputs["composite"][:, :, ::-1].copy()
    path = debug_dir / f"frame_{frame_idx:06d}.jpg"
    cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _print_eval(metrics: dict, label: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20s}: {v:.4f}")
        elif isinstance(v, dict):
            for cls, val in v.items():
                print(f"  {k}/{cls:<15s}: {val:.4f}")
        else:
            print(f"  {k:<20s}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()

    import torch
    import cv2
    import numpy as np
    from omegaconf import OmegaConf

    from fusion_perception.models.wilddet3d_wrapper import WildDet3DWrapper
    from fusion_perception.tracking.kalman_cowtracker import KalmanCoWTracker
    from fusion_perception.occupancy.bev_grid import OccupancyBEVGenerator
    from fusion_perception.tracking.trajectory_manager import SceneMemoryManager
    from fusion_perception.models.gemma_wrapper import GemmaReasoningWrapper
    from fusion_perception.pipelines.stage_runner import StageRunner
    from fusion_perception.utils.video_loader import VideoLoader
    from fusion_perception.utils.json_io import save_detections, save_tracks, save_reasoning
    from fusion_perception.utils.memory_monitor import log_gpu_memory
    from fusion_perception.segmentation.road_segmentor import RoadSegmentor

    # Validate source: need either --video or (--kitti360-root + --kitti360-seq)
    use_kitti360_frames = (
        args.video is None
        and args.kitti360_root is not None
        and args.kitti360_seq is not None
    )
    if args.video is None and not use_kitti360_frames:
        print("ERROR: provide --video or both --kitti360-root and --kitti360-seq")
        return 1

    cfg = _load_config(args.config, args.colab_config)
    run_id = args.run_id or datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(args.output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug_frames"
    if args.save_debug_frames:
        debug_dir.mkdir(exist_ok=True)

    source_label = (
        f"KITTI-360 {args.kitti360_seq}" if use_kitti360_frames else args.video
    )
    print(f"[inference] Run ID : {run_id}")
    print(f"[inference] Output : {out_dir}")
    print(f"[inference] Source : {source_label}")

    # ── Calibration + LiDAR ─────────────────────────────────────────────────
    calib = lidar_seq = None
    ego_comp = None
    kitti_root = args.kitti_root or args.kitti360_root
    if kitti_root:
        from fusion_perception.data.kitti_calibration import load_calibration, Kitti360Calib
        from fusion_perception.data.lidar_loader import KittiRawLidar, Kitti360Lidar
        try:
            calib = load_calibration(kitti_root)
            print(f"[inference] Calibration: {type(calib).__name__}")
            if args.kitti360_root:
                lidar_seq = Kitti360Lidar.from_dir(kitti_root)
            else:
                lidar_seq = KittiRawLidar.from_dir(kitti_root)
            print(f"[inference] LiDAR: {len(lidar_seq._files)} frames")
        except Exception as e:
            print(f"[inference] WARNING: LiDAR/calib load failed: {e}")

    # ── Ego-motion compensation (KITTI-360 pose-based) ───────────────────────
    if args.kitti360_root and args.kitti360_seq and isinstance(calib, Kitti360Calib):
        from fusion_perception.data.kitti360_labels import _load_poses
        from fusion_perception.tracking.ego_motion import PoseEgoCompensation
        import os as _os
        pose_file = _os.path.join(args.kitti360_root, "data_poses", args.kitti360_seq, "poses.txt")
        try:
            poses = _load_poses(pose_file)
            ego_comp = PoseEgoCompensation(poses, calib.T_pose_to_cam, calib.R_rect)
            print(f"[inference] Ego-motion: pose-based 3D ({len(poses)} pose frames)")
        except Exception as e:
            print(f"[inference] WARNING: ego-motion init failed: {e}")

    # ── GT labels for evaluation ─────────────────────────────────────────────
    gt_labels = None
    if args.eval and args.kitti360_root and args.kitti360_seq:
        from fusion_perception.data.kitti360_labels import Kitti360GTLabels
        try:
            gt_labels = Kitti360GTLabels(args.kitti360_root, args.kitti360_seq)
            print(f"[inference] GT labels: {gt_labels.annotation.num_bbox} boxes")
        except Exception as e:
            print(f"[inference] WARNING: GT label load failed: {e}")

    # ── Build models ─────────────────────────────────────────────────────────
    detector = WildDet3DWrapper(
        score_threshold=cfg.detection.score_threshold,
        fp16=cfg.detection.fp16,
    )
    detector.load(cfg.detection.checkpoint, cfg.detection.device)

    tracker = KalmanCoWTracker(
        window_size=cfg.tracking.window_size,
        max_tracks=cfg.tracking.max_tracks,
        lost_patience=cfg.tracking.lost_patience,
        confirm_age=cfg.tracking.confirm_age,
        high_score_threshold=cfg.tracking.high_score_threshold,
        low_score_threshold=cfg.tracking.low_score_threshold,
        assignment_cost_threshold=cfg.tracking.assignment_cost_threshold,
        alpha_cost=cfg.tracking.alpha_cost,
        cow_conf_threshold=cfg.tracking.cow_conf_threshold,
        min_cow_points=cfg.tracking.min_cow_points,
        velocity_decay=cfg.tracking.velocity_decay,
        ego_motion=cfg.tracking.ego_motion,
        mahal_threshold=cfg.tracking.mahal_threshold,
        lazy_cow_innovation=cfg.tracking.lazy_cow_innovation,
        device=cfg.tracking.device,
    )
    tracker.load()
    if hasattr(tracker, "cow_active"):
        if tracker.cow_active:
            print("[inference] CoWTracker loaded — hybrid KF+pixel mode active")
        else:
            print(
                "[inference] WARNING: CoWTracker failed to load — tracker is in DEGRADED KF-only mode.\n"
                "           Install: pip install git+https://github.com/facebookresearch/co-tracker"
            )

    bev_gen = OccupancyBEVGenerator(
        resolution=cfg.occupancy.resolution,
        x_range=list(cfg.occupancy.x_range),
        z_range=list(cfg.occupancy.z_range),
        decay_factor=cfg.occupancy.decay_factor,
        lidar_confidence=cfg.occupancy.lidar_confidence,
    )
    scene_mem = SceneMemoryManager()
    gemma = GemmaReasoningWrapper(
        model_id=cfg.reasoning.model_id,
        quantize_4bit=cfg.reasoning.quantize_4bit,
        max_new_tokens=cfg.reasoning.max_new_tokens,
        device=cfg.reasoning.device,
        visual_mode=False,
    )

    # ── YOLO26 detector (optional) ────────────────────────────────────────────
    yolo26 = None
    yolo26_cfg = cfg.get("yolo26", None)
    if yolo26_cfg is not None and yolo26_cfg.get("enabled", False):
        from fusion_perception.models.yolo26_detector import YOLO26Detector
        yolo26 = YOLO26Detector(
            model_id=yolo26_cfg.model_id,
            score_threshold=yolo26_cfg.score_threshold,
            device=yolo26_cfg.device,
            dynamic_coco_ids=dict(yolo26_cfg.dynamic_coco_ids),
            static_coco_ids=dict(yolo26_cfg.static_coco_ids),
        )
        yolo26.load()
        print("[inference] YOLO26 detector loaded (hybrid mode)")

    # ── Road segmentor ────────────────────────────────────────────────────────
    road_seg = RoadSegmentor(device=cfg.tracking.device)
    road_seg.load()
    print("[inference] Road segmentor loaded")

    # ── Depth pipeline (optional) ─────────────────────────────────────────────
    depth_estimator = depth_quality_gate = lidar_anchor_obj = point_cloud_acc = None
    depth_cfg = cfg.get("depth", None)
    _depth_enabled = bool(depth_cfg.enabled) if depth_cfg is not None and hasattr(depth_cfg, "enabled") else False
    print(f"[inference] Depth pipeline enabled: {_depth_enabled}")
    if _depth_enabled:
        from fusion_perception.depth import (
            MonoDepthEstimator, DepthQualityGate, LidarDepthAnchor, PointCloudAccumulator,
        )
        depth_estimator = MonoDepthEstimator(
            model_size=depth_cfg.get("model_size", "small"),
            max_depth=float(depth_cfg.get("max_depth", 80.0)),
            device=depth_cfg.get("device", "cuda"),
        )
        depth_estimator.load()
        qg_cfg = depth_cfg.get("quality_gate", None)
        if qg_cfg is not None and qg_cfg.get("enabled", True):
            depth_quality_gate = DepthQualityGate(
                use_clip_iqa=bool(qg_cfg.get("use_clip_iqa", False)),
                min_score=float(qg_cfg.get("min_score", 0.30)),
                device=depth_cfg.get("device", "cuda"),
            )
            depth_quality_gate.load()
        la_cfg = depth_cfg.get("lidar_anchor", None)
        if la_cfg is not None and la_cfg.get("enabled", True):
            lidar_anchor_obj = LidarDepthAnchor(
                min_frustum_points=int(la_cfg.get("min_frustum_points", 5)),
                global_scale=bool(la_cfg.get("global_scale", True)),
                scale_clamp=tuple(la_cfg.get("scale_clamp", [0.5, 2.0])),
            )
        rc_cfg = depth_cfg.get("reconstruction", None)
        if rc_cfg is not None and rc_cfg.get("enabled", False):
            point_cloud_acc = PointCloudAccumulator(
                rolling_frames=int(rc_cfg.get("rolling_frames", 10)),
                stride=int(rc_cfg.get("stride", 4)),
                max_depth=float(depth_cfg.get("max_depth", 80.0)),
            )
        print("[inference] Depth pipeline loaded (DA-V2 + LiDAR anchor)")

    log_gpu_memory("Pass-1 models loaded")

    fusion_iou_thr = 0.25
    if yolo26_cfg is not None:
        fusion_iou_thr = float(yolo26_cfg.get("fusion_iou_threshold", 0.25))

    runner = StageRunner(
        detector=detector, tracker=tracker, bev_generator=bev_gen,
        scene_memory=scene_mem, gemma=gemma,
        prompts=list(cfg.detection.prompts),
        reasoning_interval=cfg.reasoning.interval_frames,
        fps=30.0, visual_reasoning=False,
        road_segmentor=road_seg, road_seg_interval=5,
        yolo26_detector=yolo26,
        fusion_iou_threshold=fusion_iou_thr,
        depth_estimator=depth_estimator,
        depth_quality_gate=depth_quality_gate,
        lidar_anchor=lidar_anchor_obj,
        point_cloud_acc=point_cloud_acc,
    )

    resize_hw = list(cfg.video.resize_hw) if cfg.video.resize_hw else None

    if use_kitti360_frames:
        from fusion_perception.data.kitti360_frame_loader import Kitti360FrameLoader
        loader = Kitti360FrameLoader(
            dataset_root=args.kitti360_root,
            sequence=args.kitti360_seq,
            resize_hw=resize_hw,
            max_frames=args.max_frames,
        )
        _fps_src = loader.fps
        _total = loader.total_frames
        # Probe frame size from first image
        _probe = next(iter(loader))
        _h, _w = _probe[1].shape[:2]
        # Re-create loader (exhausted by probe)
        loader = Kitti360FrameLoader(
            dataset_root=args.kitti360_root,
            sequence=args.kitti360_seq,
            resize_hw=resize_hw,
            max_frames=args.max_frames,
        )
        K = loader.get_intrinsics() if args.fx is None else _build_intrinsics(args, _h, _w)
        if resize_hw:
            _h, _w = resize_hw
    else:
        loader = VideoLoader(
            source=args.video,
            resize_hw=resize_hw,
            max_frames=args.max_frames,
        )
        _cap = cv2.VideoCapture(args.video)
        _fps_src = _cap.get(cv2.CAP_PROP_FPS) or 10.0
        _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _cap.release()
        if resize_hw:
            _h, _w = resize_hw
        K = _build_intrinsics(args, _h, _w)

    video_writer = None
    if not args.no_video:
        # composite width: frame + square BEV + (depth panel when enabled) + text panel
        depth_panel_w = _h if depth_estimator is not None else 0
        comp_w = _w + _h + depth_panel_w + 200
        video_writer = cv2.VideoWriter(
            str(out_dir / "output.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            _fps_src,
            (comp_w, _h),
        )

    # ── Pass 1: Detection + tracking ─────────────────────────────────────────
    print(f"\n[Pass 1] detection + tracking ({_total} total frames)...")
    all_detections: dict = {}
    all_tracks: dict = {}
    pending_memories: list = []
    n_errors = 0
    last_frame_idx = -1
    FLUSH = cfg.output.flush_interval

    from fusion_perception.data.lidar_loader import load_bin

    for frame_idx, frame_rgb, meta in loader:
        try:
            lidar_pts = None
            if calib is not None and lidar_seq is not None:
                lp = lidar_seq.get_path(frame_idx)
                if lp:
                    lidar_pts = calib.velo_to_cam(load_bin(lp))

            T_ego = ego_comp.get_transform(frame_idx) if ego_comp is not None else None
            outputs = runner.run_frame(frame_rgb, frame_idx, intrinsics=K,
                                       lidar_pts=lidar_pts, T_ego=T_ego)
            all_detections[frame_idx] = outputs["detections"]
            for t in outputs["tracks"]:
                all_tracks[t.track_id] = t

            if frame_idx % cfg.reasoning.interval_frames == 0:
                pending_memories.append((frame_idx, outputs["memory"]))

            if video_writer is not None:
                video_writer.write(outputs["composite"][:, :, ::-1])

            if args.save_debug_frames and frame_idx % args.debug_every == 0:
                _save_debug_frame(outputs, frame_idx, debug_dir)

            if (frame_idx + 1) % FLUSH == 0:
                save_detections(out_dir / "detections.json", run_id,
                                args.video, list(cfg.detection.prompts), all_detections)
                save_tracks(out_dir / "tracks.json", run_id, all_tracks)
                print(f"  checkpoint @ frame {frame_idx}  "
                      f"dets={len(all_detections)}  tracks={len(all_tracks)}")

            last_frame_idx = frame_idx

        except KeyboardInterrupt:
            print("\n[inference] Interrupted by user.")
            break
        except Exception as exc:
            n_errors += 1
            print(f"  WARNING frame {frame_idx}: {exc}")
            if n_errors >= 10:
                print("[inference] Too many errors — aborting pass 1.")
                break

    if video_writer is not None:
        video_writer.release()

    save_detections(out_dir / "detections.json", run_id,
                    args.video, list(cfg.detection.prompts), all_detections)
    save_tracks(out_dir / "tracks.json", run_id, all_tracks)
    print(f"[Pass 1] done — {last_frame_idx + 1} frames, "
          f"{len(all_tracks)} tracks, {n_errors} errors")

    # ── Free GPU ──────────────────────────────────────────────────────────────
    detector.unload()
    try:
        tracker.unload()
    except AttributeError:
        pass
    if yolo26 is not None:
        yolo26.unload()
    road_seg.unload()
    if depth_estimator is not None:
        depth_estimator.unload()
    if depth_quality_gate is not None:
        depth_quality_gate.unload()
    torch.cuda.empty_cache()
    log_gpu_memory("After unloading detection models")

    # ── Pass 2: Gemma reasoning ───────────────────────────────────────────────
    if pending_memories:
        print(f"\n[Pass 2] Gemma reasoning on {len(pending_memories)} snapshots...")
        gemma.load()
        log_gpu_memory("Gemma loaded")
        from fusion_perception.memory.gemma_memory import GemmaMemoryBuffer
        gemma_mem_buf = GemmaMemoryBuffer(maxlen=5)
        all_reasoning = []
        for i, (fidx, memory) in enumerate(pending_memories):
            try:
                mem_prefix = gemma_mem_buf.format_prefix(fidx, _fps_src)
                r = gemma.reason(memory, None, trigger_reason="interval",
                                 fps=_fps_src, memory_prefix=mem_prefix)
                gemma_mem_buf.add(r, memory.active_tracks, fidx, _fps_src)
                all_reasoning.append(r)
                anomaly_str = f"  CAUTION: {r.anomalies[0]}" if r.anomalies else ""
                print(f"  [{i+1:3d}/{len(pending_memories)}] f{fidx:4d} | "
                      f"{r.latency_ms:5.0f}ms | {r.summary[:60]}{anomaly_str}")
            except Exception as exc:
                print(f"  WARNING Gemma frame {fidx}: {exc}")

        save_reasoning(out_dir / "reasoning.json", run_id,
                       cfg.reasoning.interval_frames, all_reasoning)
        gemma.unload()
        torch.cuda.empty_cache()
        print(f"[Pass 2] done — {len(all_reasoning)} reasoning outputs")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if gt_labels is not None and all_detections:
        print("\n[Evaluation] building GT dict...")
        from fusion_perception.evaluation.metrics import (
            compute_detection_metrics, compute_tracking_metrics,
        )
        gt_dict: dict = {}
        eval_frames = sorted(all_detections.keys())
        for fidx in eval_frames[::args.eval_every]:
            labs = gt_labels.to_gt_labels(fidx)
            if labs:
                gt_dict[fidx] = labs

        if gt_dict:
            # Restrict predictions to frames that have GT annotations so that
            # detections on non-GT frames don't inflate FP counts.
            eval_preds = {k: v for k, v in all_detections.items() if k in gt_dict}
            det_m = compute_detection_metrics(eval_preds, gt_dict)
            _print_eval(det_m, "Detection (BEV IoU@0.5)")

            trk_m = compute_tracking_metrics(eval_preds, gt_dict)
            _print_eval(trk_m, "Tracking (CLEAR MOT)")

            import json
            with open(out_dir / "eval_metrics.json", "w") as f:
                json.dump({"detection": det_m, "tracking": trk_m}, f, indent=2)
            print(f"\n[Evaluation] saved → {out_dir / 'eval_metrics.json'}")
        else:
            print("[Evaluation] no GT boxes found in processed frames — skipped")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Run complete")
    print(f"{'='*55}")
    print(f"  Frames processed : {last_frame_idx + 1}")
    print(f"  Unique tracks    : {len(all_tracks)}")
    print(f"  Frame errors     : {n_errors}")
    cow_status = "active" if (hasattr(tracker, "cow_active") and tracker.cow_active) else "DEGRADED (KF-only)"
    print(f"  CoWTracker       : {cow_status}")
    print(f"  Ego-motion       : {'pose-based' if ego_comp is not None else 'disabled'}")
    print(f"  Outputs          : {out_dir}")
    print(f"{'='*55}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
