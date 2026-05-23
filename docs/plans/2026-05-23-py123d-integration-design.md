# py123d Integration Design

**Date:** 2026-05-23
**Goals:**
1. Real camera calibration as first option, `estimate_intrinsics` as fallback
2. LiDAR as second sensor modality in BEV occupancy grid
3. Full benchmarking: detection mAP + MOTA/MOTP + occupancy IoU
4. Cross-dataset comparative benchmark report (KITTI vs Waymo)

---

## Architecture Overview

```
StreamingPipeline
  ├── run(video_path)                        ← unchanged
  └── run_py123d(log_dir, dataset_name)      ← new
          ↓
     _run_frame_loop(loader, intrinsics, lidar_fn)   ← shared core
          ↓
     StageRunner.run_frame(frame, frame_idx, lidar_pts)
          ↓
     BenchmarkEvaluator.update(frame_idx, detections, tracks, occupancy)
```

---

## Section 1 — Data Source Layer

**New file: `fusion_perception/data/py123d_loader.py`**

Thin wrapper around py123d's `SceneAPI`. Yields `(frame_idx, frame_rgb, meta)` — identical shape to `VideoLoader` — plus three extra methods:

```python
loader.get_intrinsics()          # → np.ndarray [3,3], real K matrix (static per camera)
loader.get_lidar(frame_idx)      # → np.ndarray [N,3], XYZ in camera coords
loader.get_gt_labels(frame_idx)  # → list[GTLabel], for benchmarking
```

**`GTLabel` dataclass** (added to `fusion_perception/utils/dataclasses.py`):
```python
@dataclass_json
@dataclass
class GTLabel:
    track_id: int
    class_name: str
    box_3d: list[float]   # [cx, cy, cz, w, h, l, ry] camera coords
```

**`StreamingPipeline` changes:**
- Extract shared `_run_frame_loop(loader, intrinsics, lidar_fn, evaluator=None)` from `run()`
- Add `run_py123d(log_dir, dataset_name)` that builds a `Py123dLoader`, calls `get_intrinsics()` once, then delegates to `_run_frame_loop()`
- `run()` passes `intrinsics=None` → `WildDet3DWrapper` falls back to `estimate_intrinsics` as today

**New files:**
- `fusion_perception/data/__init__.py`
- `fusion_perception/data/py123d_loader.py`

---

## Section 2 — LiDAR → BEV Additive Fusion

**`OccupancyBEVGenerator.update()` signature change:**

```python
def update(
    self,
    tracks: list[Track],
    frame_idx: int,
    lidar_pts: np.ndarray | None = None,   # [N,3] XYZ in camera coords
) -> OccupancyGrid:
```

**Write order inside `update()`:**
1. Decay — multiply entire grid by `decay_factor` (unchanged)
2. LiDAR pass — project each point via `world_to_grid()`; write `max(current_cell, lidar_confidence)`
3. Track pass — stamp each active track's last 3D position at `1.0`

`max()` ensures a track centroid is never overwritten downward by a LiDAR point.

**Grid value semantics:**
- `0.0–0.3` → free / decayed
- `~0.6` → LiDAR-observed, no tracked object
- `1.0` → confirmed tracked object

**`StageRunner` change:**
- `run_frame(frame, frame_idx, lidar_pts=None)` — passes `lidar_pts` through to `bev_generator.update()`
- `_run_frame_loop()` calls `loader.get_lidar(frame_idx)` when loader is `Py123dLoader`, otherwise `None`

**Config addition (`configs/default.yaml`):**
```yaml
occupancy:
  lidar_confidence: 0.6   # new field
```

No changes to `OccupancyGrid` dataclass.

---

## Section 3 — Evaluation Metrics

All evaluators live in `fusion_perception/evaluation/`.

### Detection — `compute_detection_metrics()`

- Per-frame, per-class matching of predicted `Detection3D` vs `GTLabel` using 3D IoU threshold (default 0.5)
- Outputs: precision, recall, AP per class, mAP across classes

### Tracking — `compute_tracking_metrics()`

CLEAR MOT metrics over the full log:
- **MOTA** = 1 − (FP + FN + ID switches) / GT count
- **MOTP** = mean 3D distance of matched pairs

GT track IDs from py123d label sequence. Predicted track IDs from `Track.track_id`.

### Occupancy IoU — `compute_occupancy_iou()`

GT grid derived by projecting `GTLabel.box_3d` to BEV via the existing `world_to_grid()`. Binary threshold at `0.5` on predicted grid:

```
IoU = |pred_occupied ∩ gt_occupied| / |pred_occupied ∪ gt_occupied|
```

### `BenchmarkEvaluator`

Wraps all three, accumulates per-frame results, exposes:

```python
evaluator.finalize() → BenchmarkResult(
    map=float,
    mota=float,
    motp=float,
    mean_occ_iou=float,
    per_class_ap=dict[str, float],
    per_frame_occ_iou=list[float],
)
```

`BenchmarkResult` is a JSON-serializable dataclass.

---

## Section 4 — Cross-Dataset Benchmark Runner & Report

### `BenchmarkRunner`

```python
runner = BenchmarkRunner(pipeline_cfg, datasets=["kitti-360", "waymo"], logs_per_dataset=3)
runner.run()  # writes to outputs/benchmark_YYYY-MM-DD/
```

Calls `StreamingPipeline.run_py123d()` with a `BenchmarkEvaluator` injected per log. Evaluation rides alongside the normal frame loop — no separate code path.

### Output Structure

```
outputs/benchmark_2026-05-23/
  kitti-360/
    log_0001/   detections.json, tracks.json, occupancy.json, metrics.json
    log_0002/   ...
  waymo/
    log_0001/   ...
  report.md        ← side-by-side summary
  report.json      ← machine-readable
```

### `report.md` Format

| Dataset   | mAP  | MOTA | MOTP | Occ IoU |
|-----------|------|------|------|---------|
| kitti-360 | —    | —    | —    | —       |
| waymo     | —    | —    | —    | —       |

Plus per-class AP breakdown and per-log variance to surface outlier logs.

### `configs/benchmark.yaml`

```yaml
benchmark:
  datasets: [kitti-360, waymo]
  logs_per_dataset: 3
  iou_threshold: 0.5
  output_dir: outputs/benchmark
```

---

## Complete File Change List

### New files
- `fusion_perception/data/__init__.py`
- `fusion_perception/data/py123d_loader.py`
- `fusion_perception/evaluation/__init__.py`
- `fusion_perception/evaluation/metrics.py`
- `fusion_perception/evaluation/benchmark_evaluator.py`
- `fusion_perception/evaluation/benchmark_runner.py`
- `configs/benchmark.yaml`

### Modified files
- `fusion_perception/utils/dataclasses.py` — add `GTLabel`
- `fusion_perception/occupancy/bev_grid.py` — add `lidar_pts` parameter
- `fusion_perception/pipelines/stage_runner.py` — pass `lidar_pts` through
- `fusion_perception/pipelines/streaming_pipeline.py` — add `run_py123d()`, extract `_run_frame_loop()`
- `configs/default.yaml` — add `occupancy.lidar_confidence`
