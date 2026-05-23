"""Evaluation metrics for detection, tracking, and occupancy.

All metrics work on Python-native types (lists, dicts) rather than
tensors to stay framework-agnostic and easy to test.

Detection: axis-aligned BEV IoU + per-class mAP (trapezoidal PR curve)
Tracking:  CLEAR MOT — MOTA and MOTP  (added in Task 7)
Occupancy: binary IoU between predicted and GT-derived BEV grids  (added in Task 8)
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from fusion_perception.utils.dataclasses import (
    Detection3D, GTLabel, OccupancyGrid,
)
from fusion_perception.utils.geometry import world_to_grid

# NumPy >= 2.0 renamed np.trapz -> np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# BEV IoU
# ---------------------------------------------------------------------------

def bev_iou(box_a: list[float], box_b: list[float]) -> float:
    """Axis-aligned BEV IoU between two boxes [cx,cy,cz,w,h,l,ry].

    Projects to x-z plane using w (x-extent) and l (z-extent).
    Rotation (ry) is ignored — valid approximation for small angles.
    """
    ax1 = box_a[0] - box_a[3] / 2
    ax2 = box_a[0] + box_a[3] / 2
    az1 = box_a[2] - box_a[5] / 2
    az2 = box_a[2] + box_a[5] / 2

    bx1 = box_b[0] - box_b[3] / 2
    bx2 = box_b[0] + box_b[3] / 2
    bz1 = box_b[2] - box_b[5] / 2
    bz2 = box_b[2] + box_b[5] / 2

    inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_z = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter = inter_x * inter_z

    area_a = (ax2 - ax1) * (az2 - az1)
    area_b = (bx2 - bx1) * (bz2 - bz1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Detection — mAP
# ---------------------------------------------------------------------------

def compute_detection_metrics(
    predictions: dict[int, list[Detection3D]],
    ground_truths: dict[int, list[GTLabel]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute per-class AP and mAP over all frames.

    Args:
        predictions: {frame_idx: [Detection3D, ...]}
        ground_truths: {frame_idx: [GTLabel, ...]}
        iou_threshold: minimum BEV IoU to count as a true positive

    Returns:
        {"map": float, "per_class_ap": {class_name: float}}

    Note:
        AP uses raw (non-monotone-decreasing) trapezoidal integration over the
        precision-recall curve. This is not equivalent to PASCAL VOC 11-point
        interpolation or COCO AP. Do not directly compare these numbers to those
        benchmarks.
    """
    classes: set[str] = set()
    for dets in predictions.values():
        classes.update(d.class_name for d in dets)
    for gts in ground_truths.values():
        classes.update(g.class_name for g in gts)

    per_class_ap: dict[str, float] = {}

    for cls in classes:
        all_preds: list[tuple[int, Detection3D]] = []
        for fidx, dets in predictions.items():
            for d in dets:
                if d.class_name == cls:
                    all_preds.append((fidx, d))
        all_preds.sort(key=lambda x: x[1].score, reverse=True)

        gt_per_frame: dict[int, list[GTLabel]] = defaultdict(list)
        total_gt = 0
        for fidx, gts in ground_truths.items():
            for g in gts:
                if g.class_name == cls:
                    gt_per_frame[fidx].append(g)
                    total_gt += 1

        if total_gt == 0:
            per_class_ap[cls] = 0.0
            continue

        matched: dict[int, list[bool]] = {
            fidx: [False] * len(gts)
            for fidx, gts in gt_per_frame.items()
        }

        tp_list, fp_list = [], []
        for fidx, det in all_preds:
            gts_this = gt_per_frame.get(fidx, [])
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts_this):
                iou = bev_iou(det.box_3d, gt.box_3d)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_threshold and best_j >= 0 and not matched[fidx][best_j]:
                matched[fidx][best_j] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list)
        fp_cum = np.cumsum(fp_list)
        denom = tp_cum + fp_cum
        precision = np.where(denom > 0, tp_cum / denom, 0.0)
        recall = tp_cum / total_gt

        # Prepend origin anchor so trapezoid integrates from recall=0
        recall_full = np.concatenate([[0.0], recall])
        precision_full = np.concatenate([[1.0], precision])
        per_class_ap[cls] = float(_trapz(precision_full, recall_full))

    map_val = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return {"map": map_val, "per_class_ap": per_class_ap}


# ---------------------------------------------------------------------------
# Tracking — MOTA / MOTP (CLEAR MOT)
# ---------------------------------------------------------------------------

def compute_tracking_metrics(
    predictions: dict[int, list[Detection3D]],
    ground_truths: dict[int, list[GTLabel]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute MOTA and MOTP over all frames (CLEAR MOT metrics).

    MOTA = 1 - (FP + FN + IDSW) / total_GT
    MOTP = mean BEV center distance of matched pairs (metres)

    ID switch: a GT track_id is matched to a different predicted box
    than it was in the previous frame.

    Args:
        predictions: {frame_idx: [Detection3D, ...]}
        ground_truths: {frame_idx: [GTLabel, ...]}
        iou_threshold: minimum BEV IoU to accept a match

    Returns:
        {"mota": float, "motp": float, "fp": int, "fn": int,
         "id_switches": int, "total_gt": int}

    Note:
        When `total_gt` is 0 the denominator is clamped to 1 to avoid division by zero.
    """
    total_gt = 0
    fp_total = fn_total = idsw_total = 0
    matched_distances: list[float] = []
    prev_gt_to_pred_idx: dict[int, int] = {}  # gt track_id → pred index in prev frame

    all_frames = sorted(set(list(predictions.keys()) + list(ground_truths.keys())))

    for fidx in all_frames:
        preds = predictions.get(fidx, [])
        gts = ground_truths.get(fidx, [])
        total_gt += len(gts)

        fn = len(gts)
        matched_preds: set[int] = set()
        curr_gt_to_pred: dict[int, int] = {}

        # Build IoU matrix
        iou_matrix = np.zeros((len(gts), len(preds)), dtype=np.float32)
        for gi, gt in enumerate(gts):
            for pi, pred in enumerate(preds):
                iou_matrix[gi, pi] = bev_iou(gt.box_3d, pred.box_3d)

        # Greedy matching: highest IoU first
        while True:
            if iou_matrix.size == 0:
                break
            gi, pi = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[gi, pi] < iou_threshold:
                break

            curr_gt_to_pred[gts[gi].track_id] = pi
            matched_preds.add(pi)
            fn -= 1

            # BEV centre distance for MOTP
            gc = np.array([gts[gi].box_3d[0], gts[gi].box_3d[2]])
            pc = np.array([preds[pi].centroid_3d[0], preds[pi].centroid_3d[2]])
            matched_distances.append(float(np.linalg.norm(gc - pc)))

            # ID switch detection
            if gts[gi].track_id in prev_gt_to_pred_idx:
                if prev_gt_to_pred_idx[gts[gi].track_id] != pi:
                    idsw_total += 1

            # Suppress matched row and column
            iou_matrix[gi, :] = -1.0
            iou_matrix[:, pi] = -1.0

        fp = len(preds) - len(matched_preds)
        fp_total += fp
        fn_total += fn
        prev_gt_to_pred_idx = curr_gt_to_pred

    denom = max(total_gt, 1)
    mota = 1.0 - (fp_total + fn_total + idsw_total) / denom
    motp = float(np.mean(matched_distances)) if matched_distances else 0.0  # 0.0 sentinel when no matches occur this sequence

    return {
        "mota": float(mota),
        "motp": motp,
        "fp": fp_total,
        "fn": fn_total,
        "id_switches": idsw_total,
        "total_gt": total_gt,
    }


# ---------------------------------------------------------------------------
# Occupancy IoU
# ---------------------------------------------------------------------------

def compute_occupancy_iou(
    pred_occupancy: OccupancyGrid,
    gt_labels: list[GTLabel],
    threshold: float = 0.5,
) -> float:
    """Binary IoU between predicted BEV grid and GT-derived BEV grid.

    GT grid is built by projecting gt_labels box_3d centroids using the
    same grid parameters as pred_occupancy via world_to_grid().

    Returns 1.0 if both pred and GT are empty (perfect agreement).
    GT centroids outside the grid range are silently ignored.
    """
    pred_arr = np.array(pred_occupancy.grid, dtype=np.float32)
    pred_binary = pred_arr > threshold

    gt_arr = np.zeros_like(pred_arr)
    for gt in gt_labels:
        cell = world_to_grid(
            gt.box_3d[0], gt.box_3d[2],
            pred_occupancy.x_range,
            pred_occupancy.z_range,
            pred_occupancy.resolution,
        )
        if cell is not None:
            row = min(cell[0], gt_arr.shape[0] - 1)
            col = min(cell[1], gt_arr.shape[1] - 1)
            gt_arr[row, col] = 1.0
    gt_binary = gt_arr.astype(bool)

    # Both empty → perfect agreement
    if not pred_binary.any() and not gt_binary.any():
        return 1.0

    intersection = float((pred_binary & gt_binary).sum())
    union = float((pred_binary | gt_binary).sum())
    return intersection / union if union > 0 else 0.0
