import numpy as np
from fusion_perception.evaluation.metrics import bev_iou, compute_detection_metrics, compute_tracking_metrics, compute_occupancy_iou
from fusion_perception.evaluation.benchmark_evaluator import BenchmarkEvaluator
from fusion_perception.utils.dataclasses import Detection3D, GTLabel, OccupancyGrid


def _make_det(cx, cz, w=2.0, l=4.0, class_name="car", score=0.9, frame_idx=0):
    return Detection3D(
        frame_idx=frame_idx, class_id=0, class_name=class_name,
        score=score, score_2d=score, score_3d=score,
        box_2d=[0.0, 0.0, 10.0, 10.0],
        box_3d=[cx, 0.0, cz, w, 1.5, l, 0.0],
        centroid_2d=[5.0, 5.0],
        centroid_3d=[cx, 0.0, cz],
        depth=cz,
    )


def _make_gt(cx, cz, w=2.0, l=4.0, class_name="car", track_id=1):
    return GTLabel(
        track_id=track_id,
        class_name=class_name,
        box_3d=[cx, 0.0, cz, w, 1.5, l, 0.0],
    )


def test_bev_iou_identical_boxes():
    box = [0.0, 0.0, 10.0, 1.5, 1.5, 4.0, 0.0]
    assert abs(bev_iou(box, box) - 1.0) < 1e-6


def test_bev_iou_non_overlapping_boxes():
    a = [0.0, 0.0, 0.0, 2.0, 1.5, 4.0, 0.0]
    b = [100.0, 0.0, 100.0, 2.0, 1.5, 4.0, 0.0]
    assert bev_iou(a, b) == 0.0


def test_bev_iou_partial_overlap():
    a = [0.0, 0.0, 0.0, 4.0, 1.5, 4.0, 0.0]   # x: [-2,2], z: [-2,2]
    b = [2.0, 0.0, 0.0, 4.0, 1.5, 4.0, 0.0]   # x: [0,4], z: [-2,2]
    iou = bev_iou(a, b)
    assert abs(iou - 1/3) < 1e-9


def test_compute_detection_metrics_perfect_match():
    preds = {0: [_make_det(0.0, 10.0)]}
    gts   = {0: [_make_gt(0.0, 10.0)]}
    result = compute_detection_metrics(preds, gts, iou_threshold=0.5)
    assert result["map"] == 1.0
    assert result["per_class_ap"]["car"] == 1.0


def test_compute_detection_metrics_no_predictions():
    preds = {0: []}
    gts   = {0: [_make_gt(0.0, 10.0)]}
    result = compute_detection_metrics(preds, gts, iou_threshold=0.5)
    assert result["map"] == 0.0


def test_compute_detection_metrics_no_gt():
    preds = {0: [_make_det(0.0, 10.0)]}
    gts   = {0: []}
    result = compute_detection_metrics(preds, gts, iou_threshold=0.5)
    assert result["map"] == 0.0


def test_tracking_metrics_perfect():
    pred_history = {
        0: [_make_det(0.0, 10.0)],
        1: [_make_det(0.1, 9.5)],
        2: [_make_det(0.2, 9.0)],
    }
    gt_history = {
        0: [_make_gt(0.0, 10.0, track_id=1)],
        1: [_make_gt(0.1, 9.5, track_id=1)],
        2: [_make_gt(0.2, 9.0, track_id=1)],
    }
    result = compute_tracking_metrics(pred_history, gt_history, iou_threshold=0.5)
    assert result["mota"] == 1.0
    assert result["motp"] == 0.0

def test_tracking_metrics_all_false_positives():
    pred_history = {0: [_make_det(0.0, 10.0)]}
    gt_history   = {0: []}
    result = compute_tracking_metrics(pred_history, gt_history, iou_threshold=0.5)
    assert result["mota"] <= 0.0

def test_tracking_metrics_all_misses():
    pred_history = {0: []}
    gt_history   = {0: [_make_gt(0.0, 10.0)]}
    result = compute_tracking_metrics(pred_history, gt_history, iou_threshold=0.5)
    assert result["mota"] == 0.0
    assert result["fn"] == 1
    assert result["fp"] == 0
    assert result["total_gt"] == 1
    assert result["id_switches"] == 0


def test_tracking_metrics_id_switch():
    # Frame 0: GT(track_id=1) matches pred[0] (best IoU)
    # Frame 1: pred order reversed so GT(track_id=1) matches pred[1] → 1 ID switch
    pred_history = {
        0: [_make_det(0.0, 10.0, score=0.9), _make_det(5.0, 10.0, score=0.1)],
        1: [_make_det(5.0, 10.0, score=0.9), _make_det(0.1, 10.0, score=0.1)],
    }
    gt_history = {
        0: [_make_gt(0.0, 10.0, track_id=1)],
        1: [_make_gt(0.1, 10.0, track_id=1)],
    }
    result = compute_tracking_metrics(pred_history, gt_history, iou_threshold=0.5)
    assert result["id_switches"] == 1


# ---------------------------------------------------------------------------
# Occupancy IoU tests
# ---------------------------------------------------------------------------

def _make_grid(occupied_cells: list[tuple[int, int]], rows=20, cols=40) -> OccupancyGrid:
    grid = [[0.0] * cols for _ in range(rows)]
    for r, c in occupied_cells:
        grid[r][c] = 1.0
    return OccupancyGrid(
        frame_idx=0, resolution=1.0,
        x_range=[-10.0, 10.0], z_range=[0.0, 20.0],
        grid=grid, decay_factor=0.95,
    )


def test_occupancy_iou_perfect():
    # GT centroid x=0.0, z=5.0 → col=int((0.0-(-10))/1)=10, row=int((5.0-0)/1)=5
    pred = _make_grid([(5, 10)])
    gts = [_make_gt(0.0, 5.0, w=1.0, l=1.0)]
    iou = compute_occupancy_iou(pred, gts)
    assert iou == 1.0


def test_occupancy_iou_no_overlap():
    # pred occupies (0,0); GT centroid at x=5.0, z=18.0 → col=15, row=18
    pred = _make_grid([(0, 0)])
    gts = [_make_gt(5.0, 18.0, w=1.0, l=1.0)]
    iou = compute_occupancy_iou(pred, gts)
    assert iou == 0.0


def test_occupancy_iou_empty_pred_and_gt():
    pred = _make_grid([])
    iou = compute_occupancy_iou(pred, [])
    assert iou == 1.0  # both empty = perfect agreement


def test_occupancy_iou_out_of_range_gt_ignored():
    pred = _make_grid([])
    # GT centroid at x=999 is outside x_range=[-10,10] → world_to_grid returns None → ignored
    gts = [_make_gt(999.0, 5.0, w=1.0, l=1.0)]
    iou = compute_occupancy_iou(pred, gts)
    # Both pred and gt_arr are empty → returns 1.0
    assert iou == 1.0


def test_occupancy_iou_partial_overlap():
    # pred has 2 occupied cells, GT maps to 1 of them → IoU = 1/2 = 0.5
    pred = _make_grid([(5, 10), (6, 10)])
    gts = [_make_gt(0.0, 5.0, w=1.0, l=1.0)]  # maps to (5, 10)
    iou = compute_occupancy_iou(pred, gts)
    assert abs(iou - 1/2) < 1e-6


def test_benchmark_evaluator_finalize():
    ev = BenchmarkEvaluator(dataset="kitti-360", log_id="log_0001", iou_threshold=0.5)

    det = _make_det(0.0, 10.0)
    gt  = _make_gt(0.0, 10.0, track_id=1)
    grid = _make_grid([(5, 10)])  # matches GT position

    ev.update(
        frame_idx=0,
        detections=[det],
        tracks=[],
        occupancy=grid,
        gt_labels=[gt],
    )

    result = ev.finalize()
    assert result.dataset == "kitti-360"
    assert result.log_id == "log_0001"
    assert 0.0 <= result.map <= 1.0
    assert 0.0 <= result.mean_occ_iou <= 1.0
    assert len(result.per_frame_occ_iou) == 1


def test_benchmark_evaluator_empty_finalize():
    """finalize() on zero frames returns sensible sentinel values."""
    ev = BenchmarkEvaluator(dataset="kitti-360", log_id="log_0001")
    result = ev.finalize()
    assert result.map == 0.0
    assert result.mean_occ_iou == 0.0
    assert result.per_frame_occ_iou == []


def test_benchmark_evaluator_multi_frame_occ_iou():
    """per_frame_occ_iou has correct length and mean_occ_iou is the arithmetic mean."""
    ev = BenchmarkEvaluator(dataset="kitti-360", log_id="log_0001")
    grid_match = _make_grid([(5, 10)])
    grid_miss  = _make_grid([])
    gt = _make_gt(0.0, 5.0, w=1.0, l=1.0)

    ev.update(0, [], [], grid_match, [gt])   # IoU = 1.0 (perfect)
    ev.update(1, [], [], grid_miss,  [gt])   # IoU = 0.0 (no overlap)

    result = ev.finalize()
    assert len(result.per_frame_occ_iou) == 2
    assert abs(result.mean_occ_iou - 0.5) < 1e-6

    # Verify per_frame_occ_iou is a copy, not the internal list
    result.per_frame_occ_iou.append(99.0)
    result2 = ev.finalize()
    assert len(result2.per_frame_occ_iou) == 2  # internal list unchanged
