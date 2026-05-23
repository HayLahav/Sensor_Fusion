"""Multi-dataset benchmark runner and report generator.

Usage:
    runner = BenchmarkRunner(pipeline_cfg, datasets=["kitti-360", "waymo"], logs_per_dataset=3)
    runner.run()

Or just the report generator for testing:
    md, data = generate_report(results, output_dir=Path("outputs/benchmark"))
"""
from __future__ import annotations
import json
import os
import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from omegaconf import DictConfig, OmegaConf

from fusion_perception.utils.dataclasses import BenchmarkResult
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("benchmark_runner")


def generate_report(
    results: list[BenchmarkResult],
    output_dir: Path,
) -> tuple[str, dict]:
    """Aggregate results per dataset and write report.md + report.json.

    Returns (markdown_string, report_dict).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate per dataset
    by_dataset: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in results:
        by_dataset[r.dataset].append(r)

    report: dict[str, dict] = {}
    for dataset, res_list in by_dataset.items():
        report[dataset] = {
            "map":          float(np.mean([r.map for r in res_list])),
            "map_std":      float(np.std([r.map for r in res_list])),
            "mota":         float(np.mean([r.mota for r in res_list])),
            "motp":         float(np.mean([r.motp for r in res_list])),
            "mean_occ_iou": float(np.mean([r.mean_occ_iou for r in res_list])),
            "n_logs":       len(res_list),
            "per_log": [r.to_dict() for r in res_list],
        }

    # Build markdown
    lines = [
        f"# Benchmark Report — {datetime.date.today()}",
        "",
        "## Summary",
        "",
        "| Dataset | Logs | mAP | MOTA | MOTP | Occ IoU |",
        "|---------|------|-----|------|------|---------|",
    ]
    for dataset, agg in report.items():
        lines.append(
            f"| {dataset} | {agg['n_logs']} "
            f"| {agg['map']:.3f}±{agg['map_std']:.3f} "
            f"| {agg['mota']:.3f} "
            f"| {agg['motp']:.3f} "
            f"| {agg['mean_occ_iou']:.3f} |"
        )

    lines += ["", "## Per-Log Details", ""]
    for dataset, agg in report.items():
        lines.append(f"### {dataset}")
        lines.append("")
        lines.append("| Log | mAP | MOTA | MOTP | Occ IoU |")
        lines.append("|-----|-----|------|------|---------|")
        for log in agg["per_log"]:
            lines.append(
                f"| {log['log_id']} "
                f"| {log['map']:.3f} "
                f"| {log['mota']:.3f} "
                f"| {log['motp']:.3f} "
                f"| {log['mean_occ_iou']:.3f} |"
            )
        lines.append("")

    md = "\n".join(lines)
    (output_dir / "report.md").write_text(md, encoding="utf-8")
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    logger.info(f"Report written to {output_dir}")
    return md, report


class BenchmarkRunner:
    """Runs the full pipeline on N logs per dataset and generates a report."""

    def __init__(
        self,
        pipeline_cfg: DictConfig,
        datasets: list[str],
        logs_per_dataset: int = 3,
        camera_name: str = "camera",
        output_dir: str = "outputs/benchmark",
    ) -> None:
        self.pipeline_cfg = pipeline_cfg
        self.datasets = datasets
        self.logs_per_dataset = logs_per_dataset
        self.camera_name = camera_name
        self.output_dir = Path(output_dir) / str(datetime.date.today())
        _iou = OmegaConf.select(pipeline_cfg, "benchmark.iou_threshold")
        self.iou_threshold = float(_iou) if _iou is not None else 0.5

    def run(self) -> dict:
        """Run pipeline on all datasets/logs. Returns the report dict."""
        from fusion_perception.pipelines.streaming_pipeline import StreamingPipeline
        from fusion_perception.evaluation.benchmark_evaluator import BenchmarkEvaluator

        data_root = Path(os.environ.get("PY123D_DATA_ROOT", "data"))
        all_results: list[BenchmarkResult] = []

        for dataset in self.datasets:
            dataset_dir = data_root / dataset
            if not dataset_dir.exists():
                logger.warning(f"Dataset directory not found, skipping: {dataset_dir}")
                continue
            log_dirs = sorted(dataset_dir.iterdir())[:self.logs_per_dataset]

            for log_dir in log_dirs:
                log_id = log_dir.name
                logger.info(f"Benchmarking {dataset}/{log_id}")

                run_output = self.output_dir / dataset / log_id
                run_cfg = OmegaConf.merge(self.pipeline_cfg, {"output": {"base_dir": str(run_output)}})

                pipeline = StreamingPipeline(run_cfg)
                evaluator = BenchmarkEvaluator(
                    dataset=dataset,
                    log_id=log_id,
                    iou_threshold=self.iou_threshold,
                )
                pipeline.run_py123d(
                    log_dir=str(log_dir),
                    dataset_name=dataset,
                    camera_name=self.camera_name,
                    evaluator=evaluator,
                )
                result = evaluator.finalize()

                # Save per-log metrics JSON
                run_output.mkdir(parents=True, exist_ok=True)
                (run_output / "metrics.json").write_text(
                    json.dumps(result.to_dict(), indent=2), encoding="utf-8"
                )
                all_results.append(result)
                logger.info(
                    f"{dataset}/{log_id}: mAP={result.map:.3f} "
                    f"MOTA={result.mota:.3f} OccIoU={result.mean_occ_iou:.3f}"
                )

        _, report = generate_report(all_results, self.output_dir)
        return report
