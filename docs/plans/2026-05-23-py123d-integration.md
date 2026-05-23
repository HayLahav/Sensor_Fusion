# py123d Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate py123d as a dataset data source with real camera calibration, LiDAR-augmented BEV occupancy, full detection+tracking+occupancy benchmarking, and a cross-dataset comparative report across KITTI and Waymo.

**Architecture:** `Py123dLoader` wraps py123d's `SceneAPI` and yields the same `(frame_idx, frame_rgb, meta)` tuples as `VideoLoader`. `StreamingPipeline` gains `run_py123d()` alongside `run()`, both delegating to a shared `_run_frame_loop()`. `OccupancyBEVGenerator.update()` gains an optional `lidar_pts` parameter — LiDAR stamps cells at `lidar_confidence` (0.6), track centroids at 1.0, both decay together. `BenchmarkEvaluator` rides the frame loop accumulating detection mAP, MOTA/MOTP, and occupancy IoU. `BenchmarkRunner` runs N logs per dataset and emits a markdown report.

**Tech Stack:** Python 3.10+, py123d, numpy, omegaconf, existing fusion_perception stack

---

## Prerequisites

Install py123d with dataset extras and convert logs before running:

```bash
pip install py123d[kitti-360] py123d[waymo]
export PY123D_DATA_ROOT=/path/to/data

# Convert 3 KITTI-360 validation logs
py123d-conversion dataset=kitti-360 \
  dataset.parser.splits='[kitti360_val]' \
  dataset.parser.downloader.num_logs=3

# Convert 3 Waymo logs
py123d-conversion dataset=waymo-perception \
  dataset.parser.splits='[waymo_val]' \
  dataset.parser.downloader.num_logs=3
```

---

## Phase 1 — Data Contracts

### Task 1: GTLabel and BenchmarkResult dataclasses

**Files:**
- Modify: `fusion_perception/utils/dataclasses.py`
- Modify: `tests/test_dataclasses.py`

**Step 1: Write the failing tests**

Add to `tests/test_dataclasses.py`:

```python
from fusion_perception.utils.dataclasses import GTLabel, BenchmarkResult

def test_gt_label_serializes():
    label = GTLabel(
        track_id=1,
        class_name="car",
        box_3d=[2.0, 0.5, 15.0, 1.8, 1.5, 4.2, 0.0],
    )
    d = label.to_dict()
    assert d["class_name"] == "car"
    assert d["box_3d"][2] == 15.0

def test_benchmark_result_serializes():
    result = BenchmarkResult(
        dataset="kitti-360",
        log_id="log_0001",
        map=0.42,
        mota=0.61,
        motp=0.38,
        mean_occ_iou=0.55,
        per_class_ap={"car": 0.51, "person": 0.33},
        per_frame_occ_iou=[0.5, 0.6, 0.55],
    )
    assert result.to_dict()["map"] == 0.42
    assert result.to_dict()["per_class_ap"]["car"] == 0.51
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataclasses.py::test_gt_label_serializes tests/test_dataclasses.py::test_benchmark_result_serializes -v
```

Expected: `ImportError: cannot import name 'GTLabel'`

**Step 3: Add dataclasses to `fusion_perception/utils/dataclasses.py`**

Append after `ReasoningOutput`:

```python
@dataclass_json
@dataclass
class GTLabel:
    """Ground truth label from py123d dataset."""
    track_id: int
    class_name: str
    box_3d: list[float]   # [cx, cy, cz, w, h, l, ry] camera coords


@dataclass_json
@dataclass
class BenchmarkResult:
    """Aggregated metrics for one pipeline run on one dataset log."""
    dataset: str
    log_id: str
    map: float
    mota: float
    motp: float
    mean_occ_iou: float
    per_class_ap: dict
    per_frame_occ_iou: list
```

**Step 4: Run tests**

```bash
pytest tests/test_dataclasses.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add fusion_perception/utils/dataclasses.py tests/test_dataclasses.py
git commit -m "feat: add GTLabel and BenchmarkResult dataclasses"
```

---

## Phase 2 — Data Source Layer

### Task 2: Py123dLoader

**Files:**
- Create: `fusion_perception/data/__init__.py`
- Create: `fusion_perception/data/py123d_loader.py`
- Create: `tests/test_py123d_loader.py`

**Step 1: Write the failing tests**

```python
# tests/test_py123d_loader.py
from unittest.mock import MagicMock, patch
import numpy as np
from fusion_perception.data.py123d_loader import Py123dLoader
from fusion_perception.utils.dataclasses import GTLabel


def _make_mock_scene():
    """Build a minimal mock of py123d SceneAPI."""
    scene = MagicMock()
    scene.fps = 10.0

    # Camera: iterable of (timestamp, frame_rgb)
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scene.cameras.__getitem__.return_value.frames = [
        (0.0, fake_frame),
        (0.1, fake_frame),
        (0.2, fake_frame),
    ]
    scene.cameras.__getitem__.return_value.intrinsics = np.eye(3, dtype=np.float32)

    # LiDAR: returns [N, 3] array
    scene.get_lidar_at_timestamp.return_value = np.zeros((100, 3), dtype=np.float32)

    # Labels: list of dicts per timestamp
    scene.get_labels_at_timestamp.return_value = [
        {"track_id": 1, "class_name": "car",
         "box_3d": [2.0, 0.5, 15.0, 1.8, 1.5, 4.2, 0.0]},
    ]
    return scene


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_loader_yields_frames(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    frames = list(loader)
    assert len(frames) == 3
    idx, frame, meta = frames[0]
    assert idx == 0
    assert frame.shape == (480, 640, 3)
    assert "fps" in meta
    assert "total_frames" in meta


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_intrinsics_returns_3x3(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    K = loader.get_intrinsics()
    assert K.shape == (3, 3)
    assert K.dtype == np.float32


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_lidar_returns_nx3(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    pts = loader.get_lidar(frame_idx=0)
    assert pts is not None
    assert pts.ndim == 2
    assert pts.shape[1] == 3


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_gt_labels_returns_list(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    labels = loader.get_gt_labels(frame_idx=0)
    assert isinstance(labels, list)
    assert all(isinstance(lb, GTLabel) for lb in labels)
    assert labels[0].class_name == "car"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_py123d_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'fusion_perception.data'`

**Step 3: Write `fusion_perception/data/__init__.py`**

```python
from .py123d_loader import Py123dLoader
```

**Step 4: Write `fusion_perception/data/py123d_loader.py`**

```python
"""Thin wrapper around py123d SceneAPI.

Yields (frame_idx, frame_rgb, meta) identical to VideoLoader so
StreamingPipeline._run_frame_loop() works with either source.

py123d API calls are isolated here — if py123d's API changes,
only this file needs updating. The three extra methods expose
calibration, LiDAR, and GT labels for benchmark-aware runs.

NOTE: py123d must be installed and logs pre-converted via:
  py123d-conversion dataset=<name> ...
before constructing this loader.
"""
from __future__ import annotations
import numpy as np
from typing import Iterator, Optional

try:
    from py123d import SceneAPI
except ImportError:
    raise ImportError(
        "py123d not installed. Run: pip install py123d[kitti-360]"
    )

from fusion_perception.utils.dataclasses import GTLabel
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("py123d_loader")


class Py123dLoader:
    """Iterates camera frames from a py123d-converted log.

    Yields: (frame_idx: int, frame_rgb: np.ndarray [H,W,3], meta: dict)
    """

    def __init__(self, log_dir: str, camera_name: str = "camera") -> None:
        self._scene = SceneAPI(log_dir)
        self._camera_name = camera_name
        self._cam = self._scene.cameras[camera_name]
        # Cache frame list: list of (timestamp, frame_rgb)
        self._frames: list[tuple[float, np.ndarray]] = list(self._cam.frames)
        self._fps: float = float(self._scene.fps)
        logger.info(
            f"Py123dLoader: {log_dir} | camera={camera_name} "
            f"| {len(self._frames)} frames @ {self._fps:.1f}fps"
        )

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, dict]]:
        meta = {"fps": self._fps, "total_frames": len(self._frames)}
        for frame_idx, (ts, frame_rgb) in enumerate(self._frames):
            yield frame_idx, frame_rgb, meta

    def get_intrinsics(self) -> np.ndarray:
        """Return real camera intrinsics [3,3] float32."""
        K = self._cam.intrinsics
        return np.asarray(K, dtype=np.float32)

    def get_lidar(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return synchronized LiDAR point cloud [N,3] (XYZ camera coords).

        Returns None if no LiDAR is available for this frame.
        """
        if frame_idx >= len(self._frames):
            return None
        ts = self._frames[frame_idx][0]
        try:
            pts = self._scene.get_lidar_at_timestamp(ts)
            return np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        except Exception:
            return None

    def get_gt_labels(self, frame_idx: int) -> list[GTLabel]:
        """Return ground truth labels for this frame."""
        if frame_idx >= len(self._frames):
            return []
        ts = self._frames[frame_idx][0]
        try:
            raw = self._scene.get_labels_at_timestamp(ts)
            return [
                GTLabel(
                    track_id=int(lb["track_id"]),
                    class_name=str(lb["class_name"]),
                    box_3d=[float(v) for v in lb["box_3d"]],
                )
                for lb in raw
            ]
        except Exception:
            return []
```

**Step 5: Run tests**

```bash
pytest tests/test_py123d_loader.py -v
```

Expected: all 4 PASS

**Step 6: Commit**

```bash
git add fusion_perception/data/ tests/test_py123d_loader.py
git commit -m "feat: add Py123dLoader wrapping py123d SceneAPI"
```

---

## Phase 3 — LiDAR BEV Fusion

### Task 3: BEV additive LiDAR fusion

**Files:**
- Modify: `fusion_perception/occupancy/bev_grid.py`
- Modify: `configs/default.yaml`
- Modify: `tests/test_bev_grid.py`

**Step 1: Write the failing tests**

Add to `tests/test_bev_grid.py`:

```python
import numpy as np
from fusion_perception.occupancy.bev_grid import OccupancyBEVGenerator

def test_lidar_points_stamp_at_lidar_confidence():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=1.0,
        lidar_confidence=0.6,
    )
    # LiDAR point at x=0, z=5 → col=10, row=5
    lidar_pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    grid = gen.update([], frame_idx=0, lidar_pts=lidar_pts)
    assert abs(grid.grid[5][10] - 0.6) < 1e-5

def test_track_centroid_overwrites_lidar_upward():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=1.0,
        lidar_confidence=0.6,
    )
    from fusion_perception.utils.dataclasses import Track
    track = Track(
        track_id=1, class_name="car",
        first_seen=0, last_seen=0,
        centroid_history=[[0.0, 0.0]],
        position_3d_history=[[0.0, 0.0, 5.0]],  # x=0, z=5
        cow_query_point=[0.0, 0.0],
        is_active=True, occlusion_count=0,
    )
    lidar_pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    grid = gen.update([track], frame_idx=0, lidar_pts=lidar_pts)
    # Track stamps 1.0 over lidar 0.6
    assert grid.grid[5][10] == 1.0

def test_lidar_does_not_overwrite_track_downward():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=1.0,
        lidar_confidence=0.6,
    )
    # Manually stamp a cell at 1.0 first via track, then pass lidar on same cell
    from fusion_perception.utils.dataclasses import Track
    track = Track(
        track_id=1, class_name="car",
        first_seen=0, last_seen=0,
        centroid_history=[[0.0, 0.0]],
        position_3d_history=[[0.0, 0.0, 5.0]],
        cow_query_point=[0.0, 0.0],
        is_active=True, occlusion_count=0,
    )
    lidar_pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    grid = gen.update([track], frame_idx=0, lidar_pts=lidar_pts)
    assert grid.grid[5][10] == 1.0  # max(lidar=0.6, track=1.0) = 1.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_bev_grid.py::test_lidar_points_stamp_at_lidar_confidence -v
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'lidar_confidence'`

**Step 3: Update `fusion_perception/occupancy/bev_grid.py`**

```python
"""BEV occupancy grid with temporal exponential decay.

Grid convention:
  rows  = forward (z) axis, row 0 = z_min
  cols  = lateral (x) axis, col 0 = x_min
  value = occupancy probability [0.0, 1.0]

Cell value semantics:
  0.0–0.3  → free / decayed
  ~0.6     → LiDAR-observed, no tracked object
  1.0      → confirmed tracked object

TODO: Add ray-casting free-space estimation from ego origin.
TODO: Support multi-layer grids (height slices).
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from fusion_perception.utils.dataclasses import Track, OccupancyGrid
from fusion_perception.utils.geometry import world_to_grid
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("bev_grid")


class OccupancyBEVGenerator:
    """Stateful BEV occupancy grid. Call update() once per frame."""

    def __init__(
        self,
        resolution: float,
        x_range: list[float],
        z_range: list[float],
        decay_factor: float,
        lidar_confidence: float = 0.6,
    ) -> None:
        self.resolution = resolution
        self.x_range = x_range
        self.z_range = z_range
        self.decay_factor = decay_factor
        self.lidar_confidence = lidar_confidence

        n_rows = int((z_range[1] - z_range[0]) / resolution)
        n_cols = int((x_range[1] - x_range[0]) / resolution)
        self._grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    def update(
        self,
        tracks: list[Track],
        frame_idx: int,
        lidar_pts: Optional[np.ndarray] = None,
    ) -> OccupancyGrid:
        """Apply decay, rasterize LiDAR then tracks, return updated grid."""
        # 1. Decay
        self._grid *= self.decay_factor

        # 2. LiDAR pass — use max() so track stamps can only go up
        if lidar_pts is not None and len(lidar_pts) > 0:
            for pt in lidar_pts:
                cell = world_to_grid(
                    float(pt[0]), float(pt[2]),
                    self.x_range, self.z_range, self.resolution,
                )
                if cell is not None:
                    row, col = cell
                    self._grid[row, col] = max(
                        self._grid[row, col], self.lidar_confidence
                    )

        # 3. Track pass — confirmed objects always write 1.0
        for track in tracks:
            if not track.position_3d_history:
                continue
            x, _, z = track.position_3d_history[-1]
            cell = world_to_grid(x, z, self.x_range, self.z_range, self.resolution)
            if cell is not None:
                row, col = cell
                self._grid[row, col] = 1.0

        logger.debug(
            f"Frame {frame_idx}: "
            f"{int((self._grid > 0.5).sum())} occupied cells"
        )

        return OccupancyGrid(
            frame_idx=frame_idx,
            resolution=self.resolution,
            x_range=self.x_range,
            z_range=self.z_range,
            grid=self._grid.tolist(),
            decay_factor=self.decay_factor,
        )

    def get_freespace_mask(self) -> np.ndarray:
        """Binary mask: True = free cell."""
        return self._grid <= 0.5

    def reset(self) -> None:
        self._grid[:] = 0.0
        logger.info("OccupancyBEVGenerator reset")
```

**Step 4: Add `lidar_confidence` to `configs/default.yaml`**

In the `occupancy:` section, add one line:

```yaml
occupancy:
  resolution: 0.5
  x_range: [-20.0, 20.0]
  z_range: [0.0, 50.0]
  decay_factor: 0.95
  lidar_confidence: 0.6    # ← add this line
```

**Step 5: Run tests**

```bash
pytest tests/test_bev_grid.py -v
```

Expected: all PASS (existing + 3 new)

**Step 6: Commit**

```bash
git add fusion_perception/occupancy/bev_grid.py configs/default.yaml tests/test_bev_grid.py
git commit -m "feat: add LiDAR additive layer to BEV grid with lidar_confidence"
```

---

### Task 4: StageRunner LiDAR passthrough

**Files:**
- Modify: `fusion_perception/pipelines/stage_runner.py`

**Step 1: Update `run_frame()` in `fusion_perception/pipelines/stage_runner.py`**

Change the signature and the `bev_generator.update()` call:

```python
def run_frame(
    self,
    frame: np.ndarray,
    frame_idx: int,
    intrinsics: Optional[np.ndarray] = None,
    lidar_pts: Optional[np.ndarray] = None,   # ← new
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
    occupancy: OccupancyGrid = self.bev_generator.update(
        tracks, frame_idx, lidar_pts=lidar_pts   # ← pass through
    )
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
```

Also add `np` import for the type hint — it's already imported.

**Step 2: Verify existing tests still pass**

```bash
pytest tests/ -v
```

Expected: all PASS (lidar_pts defaults to None, no behaviour change)

**Step 3: Commit**

```bash
git add fusion_perception/pipelines/stage_runner.py
git commit -m "feat: pass lidar_pts through StageRunner to BEV generator"
```

---

## Phase 4 — Pipeline Refactor

### Task 5: Extract `_run_frame_loop` and add `run_py123d`

**Files:**
- Modify: `fusion_perception/pipelines/streaming_pipeline.py`

**Step 1: Rewrite `streaming_pipeline.py`**

Replace the `run()` method and add `run_py123d()` and `_run_frame_loop()`. The `_init_models()`, `_init_video_writer()`, `_flush_outputs()` methods stay unchanged.

```python
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

    self._run_frame_loop(
        runner=runner,
        loader=loader,
        intrinsics=None,       # falls back to estimate_intrinsics inside detector
        lidar_fn=None,
        evaluator=None,
    )
    logger.info(f"Pipeline complete. Outputs in {self._output_base}")

def run_py123d(
    self,
    log_dir: str,
    dataset_name: str,
    camera_name: str = "camera",
    evaluator=None,            # optional BenchmarkEvaluator
) -> None:
    """Main frame loop for py123d-converted dataset logs."""
    from fusion_perception.data.py123d_loader import Py123dLoader

    logger.info(f"Run ID: {self.run_id}")
    logger.info(f"Processing py123d log: {log_dir} ({dataset_name})")

    runner = self._init_models()
    loader = Py123dLoader(log_dir=log_dir, camera_name=camera_name)
    intrinsics = loader.get_intrinsics()   # real calibration — no estimation fallback

    self._run_frame_loop(
        runner=runner,
        loader=loader,
        intrinsics=intrinsics,
        lidar_fn=loader.get_lidar,
        evaluator=evaluator,
    )
    logger.info(f"Pipeline complete. Outputs in {self._output_base}")

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
```

**Step 2: Verify existing tests still pass**

```bash
pytest tests/ -v
```

Expected: all PASS

**Step 3: Commit**

```bash
git add fusion_perception/pipelines/streaming_pipeline.py
git commit -m "feat: extract _run_frame_loop, add run_py123d with real calibration"
```

---

## Phase 5 — Evaluation Metrics

### Task 6: Detection metrics (BEV IoU + mAP)

**Files:**
- Create: `fusion_perception/evaluation/__init__.py`
- Create: `fusion_perception/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import numpy as np
from fusion_perception.evaluation.metrics import bev_iou, compute_detection_metrics
from fusion_perception.utils.dataclasses import Detection3D, GTLabel


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
    assert 0.0 < iou < 1.0


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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ModuleNotFoundError: No module named 'fusion_perception.evaluation'`

**Step 3: Write `fusion_perception/evaluation/__init__.py`**

```python
from .metrics import bev_iou, compute_detection_metrics, compute_tracking_metrics, compute_occupancy_iou
from .benchmark_evaluator import BenchmarkEvaluator
```

**Step 4: Write detection functions in `fusion_perception/evaluation/metrics.py`**

```python
"""Evaluation metrics for detection, tracking, and occupancy.

All metrics work on Python-native types (lists, dicts) rather than
tensors to stay framework-agnostic and easy to test.

Detection: axis-aligned BEV IoU + per-class mAP (trapezoidal PR curve)
Tracking:  CLEAR MOT — MOTA and MOTP
Occupancy: binary IoU between predicted and GT-derived BEV grids
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from typing import Optional
from fusion_perception.utils.dataclasses import (
    Detection3D, GTLabel, Track, OccupancyGrid,
)
from fusion_perception.utils.geometry import world_to_grid


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
    """
    # Collect all class names
    classes: set[str] = set()
    for dets in predictions.values():
        classes.update(d.class_name for d in dets)
    for gts in ground_truths.values():
        classes.update(g.class_name for g in gts)

    per_class_ap: dict[str, float] = {}

    for cls in classes:
        # All predictions for this class, sorted by score descending
        all_preds: list[tuple[int, Detection3D]] = []
        for fidx, dets in predictions.items():
            for d in dets:
                if d.class_name == cls:
                    all_preds.append((fidx, d))
        all_preds.sort(key=lambda x: x[1].score, reverse=True)

        # GT count per frame for this class
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
        precision = tp_cum / (tp_cum + fp_cum + 1e-9)
        recall = tp_cum / total_gt

        # Trapezoidal area under PR curve
        per_class_ap[cls] = float(np.trapz(precision, recall)) if len(recall) > 1 else 0.0

    map_val = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    return {"map": map_val, "per_class_ap": per_class_ap}
```

**Step 5: Run detection metric tests**

```bash
pytest tests/test_metrics.py::test_bev_iou_identical_boxes \
       tests/test_metrics.py::test_bev_iou_non_overlapping_boxes \
       tests/test_metrics.py::test_bev_iou_partial_overlap \
       tests/test_metrics.py::test_compute_detection_metrics_perfect_match \
       tests/test_metrics.py::test_compute_detection_metrics_no_predictions \
       tests/test_metrics.py::test_compute_detection_metrics_no_gt -v
```

Expected: all 6 PASS

**Step 6: Commit**

```bash
git add fusion_perception/evaluation/__init__.py fusion_perception/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add BEV IoU and detection mAP metrics"
```

---

### Task 7: Tracking metrics (MOTA / MOTP)

**Files:**
- Modify: `fusion_perception/evaluation/metrics.py`
- Modify: `tests/test_metrics.py`

**Step 1: Write the failing tests**

Add to `tests/test_metrics.py`:

```python
from fusion_perception.evaluation.metrics import compute_tracking_metrics

def test_tracking_metrics_perfect():
    # 3 frames, 1 GT track, predictions always match
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
    # Need track_id on detections — use frame_idx trick
    result = compute_tracking_metrics(pred_history, gt_history, iou_threshold=0.5)
    assert result["mota"] == 1.0
    assert result["motp"] >= 0.0

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
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py::test_tracking_metrics_perfect -v
```

Expected: `ImportError` for `compute_tracking_metrics`

**Step 3: Add tracking metrics to `fusion_perception/evaluation/metrics.py`**

Append after `compute_detection_metrics`:

```python
# ---------------------------------------------------------------------------
# Tracking — MOTA / MOTP (CLEAR MOT)
# ---------------------------------------------------------------------------

def compute_tracking_metrics(
    predictions: dict[int, list[Detection3D]],
    ground_truths: dict[int, list[GTLabel]],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute MOTA and MOTP over all frames.

    MOTA = 1 - (FP + FN + IDSW) / total_GT
    MOTP = mean BEV center distance of matched pairs (metres)

    ID switches are counted when a GT track_id is assigned a different
    predicted box (by BEV IoU matching) than in the previous frame.
    """
    total_gt = 0
    fp_total = fn_total = idsw_total = 0
    matched_distances: list[float] = []

    prev_gt_to_pred: dict[int, int] = {}  # gt track_id → matched pred index

    all_frames = sorted(set(list(predictions.keys()) + list(ground_truths.keys())))

    for fidx in all_frames:
        preds = predictions.get(fidx, [])
        gts = ground_truths.get(fidx, [])
        total_gt += len(gts)
        fn = len(gts)
        fp = 0
        curr_gt_to_pred: dict[int, int] = {}

        matched_preds: set[int] = set()

        # Greedy matching: highest IoU first
        iou_matrix = np.zeros((len(gts), len(preds)))
        for gi, gt in enumerate(gts):
            for pi, pred in enumerate(preds):
                iou_matrix[gi, pi] = bev_iou(gt.box_3d, pred.box_3d)

        while True:
            if iou_matrix.size == 0:
                break
            best = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            gi, pi = best
            if iou_matrix[gi, pi] < iou_threshold:
                break
            curr_gt_to_pred[gts[gi].track_id] = pi
            matched_preds.add(pi)
            fn -= 1

            # BEV center distance for MOTP
            gc = [gts[gi].box_3d[0], gts[gi].box_3d[2]]
            pc = [preds[pi].centroid_3d[0], preds[pi].centroid_3d[2]]
            dist = float(np.linalg.norm(np.array(gc) - np.array(pc)))
            matched_distances.append(dist)

            # Check ID switch
            if gts[gi].track_id in prev_gt_to_pred:
                if prev_gt_to_pred[gts[gi].track_id] != pi:
                    idsw_total += 1

            iou_matrix[gi, :] = -1
            iou_matrix[:, pi] = -1

        fp = len(preds) - len(matched_preds)
        fp_total += fp
        fn_total += fn
        prev_gt_to_pred = curr_gt_to_pred

    denom = max(total_gt, 1)
    mota = 1.0 - (fp_total + fn_total + idsw_total) / denom
    motp = float(np.mean(matched_distances)) if matched_distances else 0.0

    return {
        "mota": float(mota),
        "motp": motp,
        "fp": fp_total,
        "fn": fn_total,
        "id_switches": idsw_total,
        "total_gt": total_gt,
    }
```

**Step 4: Run tracking tests**

```bash
pytest tests/test_metrics.py -k "tracking" -v
```

Expected: all 3 PASS

**Step 5: Commit**

```bash
git add fusion_perception/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add MOTA/MOTP tracking metrics"
```

---

### Task 8: Occupancy IoU metric

**Files:**
- Modify: `fusion_perception/evaluation/metrics.py`
- Modify: `tests/test_metrics.py`

**Step 1: Write the failing tests**

Add to `tests/test_metrics.py`:

```python
from fusion_perception.evaluation.metrics import compute_occupancy_iou
from fusion_perception.utils.dataclasses import OccupancyGrid

def _make_grid(occupied_cells: list[tuple[int,int]], rows=20, cols=40) -> OccupancyGrid:
    grid = [[0.0] * cols for _ in range(rows)]
    for r, c in occupied_cells:
        grid[r][c] = 1.0
    return OccupancyGrid(
        frame_idx=0, resolution=1.0,
        x_range=[-10.0, 10.0], z_range=[0.0, 20.0],
        grid=grid, decay_factor=0.95,
    )

def test_occupancy_iou_perfect():
    pred = _make_grid([(5, 10)])
    gts = [_make_gt(0.0, 5.0, w=1.0, l=1.0)]  # x=0→col10, z=5→row5
    iou = compute_occupancy_iou(pred, gts)
    assert iou == 1.0

def test_occupancy_iou_no_overlap():
    pred = _make_grid([(0, 0)])
    gts = [_make_gt(5.0, 18.0, w=1.0, l=1.0)]
    iou = compute_occupancy_iou(pred, gts)
    assert iou == 0.0

def test_occupancy_iou_empty_pred_and_gt():
    pred = _make_grid([])
    iou = compute_occupancy_iou(pred, [])
    assert iou == 1.0  # both empty = perfect agreement
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py::test_occupancy_iou_perfect -v
```

Expected: `ImportError` for `compute_occupancy_iou`

**Step 3: Add occupancy IoU to `fusion_perception/evaluation/metrics.py`**

Append after `compute_tracking_metrics`:

```python
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
    same grid parameters as pred_occupancy.
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
            gt_arr[cell[0], cell[1]] = 1.0
    gt_binary = gt_arr > threshold

    # Both empty → perfect agreement
    if not pred_binary.any() and not gt_binary.any():
        return 1.0

    intersection = float((pred_binary & gt_binary).sum())
    union = float((pred_binary | gt_binary).sum())
    return intersection / union if union > 0 else 0.0
```

**Step 4: Run occupancy IoU tests**

```bash
pytest tests/test_metrics.py -k "occupancy" -v
```

Expected: all 3 PASS

**Step 5: Run all metric tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: all PASS

**Step 6: Commit**

```bash
git add fusion_perception/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add occupancy IoU metric using GT-projected BEV grid"
```

---

### Task 9: BenchmarkEvaluator

**Files:**
- Create: `fusion_perception/evaluation/benchmark_evaluator.py`
- Modify: `tests/test_metrics.py`

**Step 1: Write the failing test**

Add to `tests/test_metrics.py`:

```python
from fusion_perception.evaluation.benchmark_evaluator import BenchmarkEvaluator

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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py::test_benchmark_evaluator_finalize -v
```

Expected: `ImportError`

**Step 3: Write `fusion_perception/evaluation/benchmark_evaluator.py`**

```python
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
        """Accumulate one frame of predictions and GT."""
        self._pred_detections[frame_idx] = detections
        self._gt_labels[frame_idx] = gt_labels
        self._occ_ious.append(
            compute_occupancy_iou(occupancy, gt_labels, threshold=0.5)
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
            per_frame_occ_iou=self._occ_ious,
        )
```

**Step 4: Run test**

```bash
pytest tests/test_metrics.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add fusion_perception/evaluation/benchmark_evaluator.py tests/test_metrics.py
git commit -m "feat: add BenchmarkEvaluator accumulating all three metric families"
```

---

## Phase 6 — Benchmark Runner

### Task 10: BenchmarkRunner, report generation, and benchmark config

**Files:**
- Create: `fusion_perception/evaluation/benchmark_runner.py`
- Create: `configs/benchmark.yaml`
- Create: `tests/test_benchmark_runner.py`

**Step 1: Write the failing test**

```python
# tests/test_benchmark_runner.py
from unittest.mock import MagicMock, patch
from fusion_perception.evaluation.benchmark_runner import generate_report
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
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_benchmark_runner.py -v
```

Expected: `ImportError`

**Step 3: Write `fusion_perception/evaluation/benchmark_runner.py`**

```python
"""Multi-dataset benchmark runner and report generator.

Usage:
    runner = BenchmarkRunner(pipeline_cfg, datasets=["kitti-360", "waymo"], logs_per_dataset=3)
    runner.run()

Or just the report generator for testing:
    md, data = generate_report(results, output_dir=Path("outputs/benchmark"))
"""
from __future__ import annotations
import json
import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from omegaconf import DictConfig

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
        self.iou_threshold = pipeline_cfg.get("benchmark", {}).get("iou_threshold", 0.5)

    def run(self) -> dict:
        """Run pipeline on all datasets/logs. Returns the report dict."""
        import os
        from fusion_perception.pipelines.streaming_pipeline import StreamingPipeline
        from fusion_perception.evaluation.benchmark_evaluator import BenchmarkEvaluator

        data_root = Path(os.environ.get("PY123D_DATA_ROOT", "data"))
        all_results: list[BenchmarkResult] = []

        for dataset in self.datasets:
            dataset_dir = data_root / dataset
            log_dirs = sorted(dataset_dir.iterdir())[:self.logs_per_dataset]

            for log_dir in log_dirs:
                log_id = log_dir.name
                logger.info(f"Benchmarking {dataset}/{log_id}")

                run_output = self.output_dir / dataset / log_id
                cfg_override = dict(self.pipeline_cfg)
                cfg_override["output"] = dict(self.pipeline_cfg.output)
                cfg_override["output"]["base_dir"] = str(run_output)

                from omegaconf import OmegaConf
                run_cfg = OmegaConf.create(cfg_override)

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
```

**Step 4: Write `configs/benchmark.yaml`**

```yaml
benchmark:
  datasets: [kitti-360, waymo]
  logs_per_dataset: 3
  iou_threshold: 0.5
  camera_name: camera
  output_dir: outputs/benchmark
```

**Step 5: Run test**

```bash
pytest tests/test_benchmark_runner.py -v
```

Expected: all PASS

**Step 6: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all PASS

**Step 7: Commit**

```bash
git add fusion_perception/evaluation/benchmark_runner.py \
        fusion_perception/evaluation/__init__.py \
        configs/benchmark.yaml \
        tests/test_benchmark_runner.py
git commit -m "feat: add BenchmarkRunner with multi-dataset report generation"
```

---

## Running a Benchmark

After converting logs (see Prerequisites):

```python
from omegaconf import OmegaConf
from fusion_perception.evaluation.benchmark_runner import BenchmarkRunner

cfg = OmegaConf.load("configs/default.yaml")
runner = BenchmarkRunner(
    pipeline_cfg=cfg,
    datasets=["kitti-360", "waymo"],
    logs_per_dataset=3,
)
report = runner.run()
# Report written to outputs/benchmark/<date>/report.md
```
