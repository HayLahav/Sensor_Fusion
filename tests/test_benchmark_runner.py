from unittest.mock import MagicMock, patch
from fusion_perception.evaluation.benchmark_runner import generate_report, BenchmarkRunner
from fusion_perception.utils.dataclasses import BenchmarkResult


def _make_result(dataset, log_id, map_val, mota, motp, occ_iou):
    return BenchmarkResult(
        dataset=dataset, log_id=log_id,
        map=map_val, mota=mota, motp=motp,
        mean_occ_iou=occ_iou,
        per_class_ap={"car": map_val},
        per_frame_occ_iou=[occ_iou],
    )


def test_generate_report_contains_datasets(tmp_path):
    results = [
        _make_result("kitti-360", "log_0001", 0.42, 0.61, 0.38, 0.55),
        _make_result("kitti-360", "log_0002", 0.38, 0.57, 0.41, 0.50),
        _make_result("waymo",     "log_0001", 0.35, 0.54, 0.40, 0.48),
    ]
    md, report = generate_report(results, output_dir=tmp_path)

    assert "kitti-360" in md
    assert "waymo" in md
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()

    assert "kitti-360" in report
    assert abs(report["kitti-360"]["map"] - 0.40) < 0.01  # mean of 0.42, 0.38


def test_generate_report_empty_results(tmp_path):
    """generate_report with no results produces valid (empty) output."""
    md, report = generate_report([], output_dir=tmp_path)
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "report.json").exists()
    assert report == {}


def test_generate_report_dict_structure(tmp_path):
    """report dict has all expected aggregate keys."""
    results = [
        _make_result("kitti-360", "log_0001", 0.42, 0.61, 0.38, 0.55),
    ]
    md, report = generate_report(results, output_dir=tmp_path)
    agg = report["kitti-360"]
    for key in ("map", "map_std", "mota", "motp", "mean_occ_iou", "n_logs", "per_log"):
        assert key in agg, f"Missing key: {key}"
    assert agg["n_logs"] == 1
    assert len(agg["per_log"]) == 1


def test_benchmark_runner_skips_missing_dataset_dir(tmp_path):
    """BenchmarkRunner.run() skips datasets whose directories don't exist."""
    from omegaconf import OmegaConf
    # Minimal config stub — BenchmarkRunner.run() needs a config that
    # StreamingPipeline can consume, but here we're only testing the
    # directory-missing guard path; run() will skip before touching StreamingPipeline.
    cfg = OmegaConf.create({
        "output": {"base_dir": str(tmp_path), "flush_interval": 100, "save_video": False, "video_fps": None},
        "video": {"source": "", "max_frames": None, "resize_hw": [480, 640]},
        "run_id": "test",
        "logging": {"level": "INFO", "log_file": None, "use_rich": False},
        "detection": {"checkpoint": "", "prompts": [], "score_threshold": 0.4, "fp16": False, "device": "cpu"},
        "tracking": {"backend": "kalman_cow", "window_size": 8, "max_tracks": 50, "device": "cpu",
                     "lost_patience": 30, "confirm_age": 3, "high_score_threshold": 0.5,
                     "low_score_threshold": 0.2, "assignment_cost_threshold": 0.5,
                     "alpha_cost": 0.35, "cow_conf_threshold": 0.85, "min_cow_points": 4,
                     "velocity_decay": 0.9, "lazy_cow_innovation": 0.3, "ego_motion": False,
                     "mahal_threshold": 9.21},
        "occupancy": {"resolution": 0.5, "x_range": [-20.0, 20.0], "z_range": [0.0, 50.0],
                      "decay_factor": 0.95, "lidar_confidence": 0.6},
        "reasoning": {"enabled": False, "interval_frames": 30, "visual_mode": False,
                      "model_id": "google/gemma-4-2b-it", "quantize_4bit": False,
                      "max_new_tokens": 64, "device": "cpu"},
    })
    import os
    runner = BenchmarkRunner(
        pipeline_cfg=cfg,
        datasets=["nonexistent-dataset"],
        output_dir=str(tmp_path),
    )
    # PY123D_DATA_ROOT points to tmp_path which has no "nonexistent-dataset" subdir
    old_env = os.environ.get("PY123D_DATA_ROOT")
    os.environ["PY123D_DATA_ROOT"] = str(tmp_path)
    try:
        report = runner.run()
        assert report == {}   # no datasets processed, empty report
    finally:
        if old_env is None:
            os.environ.pop("PY123D_DATA_ROOT", None)
        else:
            os.environ["PY123D_DATA_ROOT"] = old_env
