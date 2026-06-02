"""Per-log accumulator for all three metric families.

Usage:
    ev = BenchmarkEvaluator(dataset="kitti-360", log_id="log_0001")
    for frame in ...:
        ev.update(frame_idx, detections, tracks, occupancy, gt_labels)
    result = ev.finalize()   # BenchmarkResult
"""
from __future__ import annotations
from fusion_perception.utils.dataclasses import (
    Detection3D, GTLabel, Track, OccupancyGrid, BenchmarkResult,
)
from fusion_perception.evaluation.metrics import (
    compute_detection_metrics,
    compute_tracking_metrics,
    compute_occupancy_iou,
)


class BenchmarkEvaluator:
    """Accumulates per-frame ground truth and predictions, computes metrics on finalize()."""

    def __init__(
        self,
        dataset: str,
        log_id: str,
        iou_threshold: float = 0.5,
    ) -> None:
        self.dataset = dataset
        self.log_id = log_id
        self.iou_threshold = iou_threshold

        self._pred_detections: dict[int, list[Detection3D]] = {}
        self._gt_labels: dict[int, list[GTLabel]] = {}
        self._occ_ious: list[float] = []

    def update(
        self,
        frame_idx: int,
        detections: list[Detection3D],
        tracks: list[Track],
        occupancy: OccupancyGrid,
        gt_labels: list[GTLabel],
    ) -> None:
        """Accumulate one frame of predictions and GT.

        Args:
            frame_idx: Frame sequence number. Must be unique per evaluator instance.
            detections: Per-frame Detection3D predictions. Used for both
                detection and tracking metric computation.
            tracks: Active Track objects. Currently unused — tracking metrics
                are computed via CLEAR MOT matching of detections against GT boxes.
            occupancy: Predicted BEV occupancy grid for this frame.
            gt_labels: Ground truth labels for this frame.
        """
        if frame_idx in self._pred_detections:
            raise ValueError(f"frame_idx {frame_idx} already accumulated. BenchmarkEvaluator does not support overwriting frames.")
        self._pred_detections[frame_idx] = detections
        self._gt_labels[frame_idx] = gt_labels
        self._occ_ious.append(
            compute_occupancy_iou(occupancy, gt_labels, threshold=0.5)  # threshold=0.5: binary occupancy cutoff, distinct from self.iou_threshold (box matching)
        )

    def finalize(self) -> BenchmarkResult:
        """Compute all metrics and return a BenchmarkResult."""
        det_metrics = compute_detection_metrics(
            self._pred_detections, self._gt_labels, self.iou_threshold
        )
        trk_metrics = compute_tracking_metrics(
            self._pred_detections, self._gt_labels, self.iou_threshold
        )
        mean_occ_iou = (
            sum(self._occ_ious) / len(self._occ_ious) if self._occ_ious else 0.0
        )

        return BenchmarkResult(
            dataset=self.dataset,
            log_id=self.log_id,
            map=det_metrics["map"],
            mota=trk_metrics["mota"],
            motp=trk_metrics["motp"],
            mean_occ_iou=mean_occ_iou,
            per_class_ap=det_metrics["per_class_ap"],
            per_frame_occ_iou=list(self._occ_ious),
        )
