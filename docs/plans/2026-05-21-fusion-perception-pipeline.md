# Fusion Perception Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform a KITTI sensor fusion notebook into a modular, research-grade streaming perception pipeline combining WildDet3D 3D detection, CoWTracker dense point tracking (centroid anchoring), BEV occupancy estimation, and Gemma 4 2B semantic reasoning.

**Architecture:** All models stay resident in GPU (streaming frame-by-frame). WildDet3D detects 3D boxes per frame; CoWTracker tracks their centroid 2D positions across frames using its Windowed variant; occupancy BEV grid accumulates with temporal decay; Gemma runs every K frames on structured scene memory. Every stage emits typed dataclasses serialized to JSON.

**Tech Stack:** Python 3.10+, PyTorch 2.5.1+cu121, vis4d 1.0.0, CoWTracker (facebookresearch), Gemma 4 2B via transformers + bitsandbytes INT4, omegaconf YAML, OpenCV, numpy, dataclasses-json, rich

---

## Phase 1 — Foundation (no GPU needed)

### Task 1: Project scaffold & editable install

**Files:**
- Create: `setup.py`
- Create: `fusion_perception/__init__.py`
- Create: `requirements.txt`

**Step 1: Write `setup.py`**

```python
from setuptools import setup, find_packages

setup(
    name="fusion_perception",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
```

**Step 2: Write `fusion_perception/__init__.py`**

```python
"""Fusion Perception — streaming multimodal perception pipeline."""
__version__ = "0.1.0"
```

**Step 3: Write `requirements.txt`**

```
torch==2.5.1
torchvision==0.20.1
omegaconf>=2.3.0
opencv-python>=4.9.0
Pillow>=10.0.0
numpy>=1.26.0
dataclasses-json>=0.6.0
rich>=13.0.0
huggingface_hub>=0.26.0
transformers>=4.47.0
bitsandbytes>=0.44.0
accelerate>=1.2.0
```

**Step 4: Install editable**

```bash
pip install -e .
```

Expected: `Successfully installed fusion-perception-0.1.0`

**Step 5: Commit**

```bash
git add setup.py fusion_perception/__init__.py requirements.txt
git commit -m "feat: scaffold project with editable install"
```

---

### Task 2: Shared dataclasses

**Files:**
- Create: `fusion_perception/utils/dataclasses.py`
- Create: `fusion_perception/utils/__init__.py`
- Create: `tests/test_dataclasses.py`

**Step 1: Write the failing test**

```python
# tests/test_dataclasses.py
from fusion_perception.utils.dataclasses import (
    Detection3D, Track, OccupancyGrid, SceneMemory, ReasoningOutput
)
import json

def test_detection3d_serializes_to_dict():
    det = Detection3D(
        frame_idx=0, class_id=0, class_name="car", score=0.9,
        score_2d=0.88, score_3d=0.92,
        box_2d=[10.0, 20.0, 100.0, 80.0],
        box_3d=[2.1, 0.8, 15.3, 1.8, 1.5, 4.2, 0.05],
        centroid_2d=[55.0, 50.0],
        centroid_3d=[2.1, 0.8, 15.3],
        depth=15.3,
    )
    d = det.to_dict()
    assert d["class_name"] == "car"
    assert d["depth"] == 15.3

def test_track_json_roundtrip():
    track = Track(
        track_id=1, class_name="car",
        first_seen=0, last_seen=5,
        centroid_history=[[55.0, 50.0], [56.0, 49.0]],
        position_3d_history=[[2.1, 0.8, 15.3]],
        cow_query_point=[55.0, 50.0],
        is_active=True, occlusion_count=0,
    )
    json_str = track.to_json()
    restored = Track.from_json(json_str)
    assert restored.track_id == 1
    assert restored.class_name == "car"

def test_occupancy_grid_shape():
    grid = OccupancyGrid(
        frame_idx=0, resolution=0.5,
        x_range=[-20.0, 20.0], z_range=[0.0, 50.0],
        grid=[[0.0] * 80 for _ in range(100)],
        decay_factor=0.95,
    )
    assert len(grid.grid) == 100
    assert len(grid.grid[0]) == 80
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_dataclasses.py -v
```

Expected: `ImportError: cannot import name 'Detection3D'`

**Step 3: Implement `fusion_perception/utils/dataclasses.py`**

```python
"""Shared dataclasses for the fusion perception pipeline.

All types are JSON-serializable via dataclasses-json.
These are the contracts between every pipeline stage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import Optional


@dataclass_json
@dataclass
class Detection3D:
    """Single 3D detection from WildDet3D for one frame."""
    frame_idx: int
    class_id: int
    class_name: str
    score: float
    score_2d: float
    score_3d: float
    box_2d: list[float]          # [x1, y1, x2, y2] pixels
    box_3d: list[float]          # [cx, cy, cz, w, h, l, ry] camera coords
    centroid_2d: list[float]     # [x, y] pixels — center of box_2d
    centroid_3d: list[float]     # [x, y, z] camera coords
    depth: float                 # estimated depth in meters


@dataclass_json
@dataclass
class Track:
    """Persistent object identity across frames."""
    track_id: int
    class_name: str
    first_seen: int
    last_seen: int
    centroid_history: list[list[float]]        # [[x,y], ...] pixels per frame
    position_3d_history: list[list[float]]     # [[x,y,z], ...] camera coords
    cow_query_point: list[float]               # current CoWTracker query [x, y]
    is_active: bool
    occlusion_count: int                       # frames since last matched detection


@dataclass_json
@dataclass
class OccupancyGrid:
    """BEV occupancy grid state at a single frame."""
    frame_idx: int
    resolution: float               # meters per cell
    x_range: list[float]            # [x_min, x_max] meters
    z_range: list[float]            # [z_min, z_max] meters (forward)
    grid: list[list[float]]         # 2D array: 0.0=free, 1.0=occupied
    decay_factor: float


@dataclass_json
@dataclass
class SceneMemory:
    """Accumulated perception state across all frames up to now."""
    frame_idx: int
    active_tracks: list[Track]
    occupancy_grid: OccupancyGrid
    event_flags: list[str]          # e.g. ["sudden_stop:2", "new_object:5"]
    frame_count: int
    elapsed_seconds: float


@dataclass_json
@dataclass
class ReasoningOutput:
    """Gemma reasoning result for a given frame window."""
    frame_idx: int
    trigger_reason: str             # "interval" | "event"
    visual_mode: bool
    prompt_used: str
    summary: str
    anomalies: list[str]
    trajectory_nl: str
    raw_response: str
    latency_ms: float
```

**Step 4: Write `fusion_perception/utils/__init__.py`**

```python
from .dataclasses import (
    Detection3D, Track, OccupancyGrid, SceneMemory, ReasoningOutput
)
```

**Step 5: Run tests**

```bash
pytest tests/test_dataclasses.py -v
```

Expected: all 3 PASS

**Step 6: Commit**

```bash
git add fusion_perception/utils/ tests/test_dataclasses.py
git commit -m "feat: add shared dataclasses with JSON serialization"
```

---

### Task 3: Logging setup

**Files:**
- Create: `fusion_perception/utils/logging_setup.py`
- Create: `tests/test_logging_setup.py`

**Step 1: Write failing test**

```python
# tests/test_logging_setup.py
import logging
from fusion_perception.utils.logging_setup import get_logger, setup_logging

def test_get_logger_returns_named_logger():
    logger = get_logger("test_module")
    assert logger.name == "fusion_perception.test_module"

def test_setup_logging_sets_level():
    setup_logging(level="DEBUG", log_file=None)
    root = logging.getLogger("fusion_perception")
    assert root.level == logging.DEBUG
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_logging_setup.py -v
```

**Step 3: Implement `fusion_perception/utils/logging_setup.py`**

```python
"""Structured logging configuration for the pipeline."""
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> None:
    """Configure root logger for fusion_perception namespace."""
    root = logging.getLogger("fusion_perception")
    root.setLevel(getattr(logging, level.upper()))
    root.handlers.clear()

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    datefmt = "%H:%M:%S"

    if use_rich and _RICH_AVAILABLE:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the fusion_perception namespace."""
    return logging.getLogger(f"fusion_perception.{name}")
```

**Step 4: Run tests**

```bash
pytest tests/test_logging_setup.py -v
```

Expected: 2 PASS

**Step 5: Commit**

```bash
git add fusion_perception/utils/logging_setup.py tests/test_logging_setup.py
git commit -m "feat: add structured logging with rich support"
```

---

### Task 4: Geometry utilities

**Files:**
- Create: `fusion_perception/utils/geometry.py`
- Create: `tests/test_geometry.py`

**Step 1: Write failing test**

```python
# tests/test_geometry.py
import numpy as np
from fusion_perception.utils.geometry import (
    box2d_centroid, box3d_centroid, camera_to_bev,
    estimate_intrinsics,
)

def test_box2d_centroid():
    cx, cy = box2d_centroid([10.0, 20.0, 100.0, 80.0])
    assert cx == 55.0
    assert cy == 50.0

def test_box3d_centroid():
    xyz = box3d_centroid([2.1, 0.8, 15.3, 1.8, 1.5, 4.2, 0.05])
    assert xyz == [2.1, 0.8, 15.3]

def test_camera_to_bev_forward_maps_to_positive_z():
    bev_x, bev_z = camera_to_bev(x_cam=1.0, z_cam=10.0)
    assert bev_z == 10.0
    assert bev_x == 1.0

def test_estimate_intrinsics_shape():
    K = estimate_intrinsics(h=480, w=640)
    assert K.shape == (3, 3)
    assert K[0, 2] == 320.0  # cx = w/2
    assert K[1, 2] == 240.0  # cy = h/2
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_geometry.py -v
```

**Step 3: Implement `fusion_perception/utils/geometry.py`**

```python
"""Geometric utilities: 2D/3D box math, BEV projection, intrinsics."""
import numpy as np


def box2d_centroid(box: list[float]) -> tuple[float, float]:
    """Return pixel centroid [cx, cy] of a [x1,y1,x2,y2] box."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box3d_centroid(box: list[float]) -> list[float]:
    """Return [cx, cy, cz] from a [cx,cy,cz,w,h,l,ry] 3D box."""
    return box[:3]


def camera_to_bev(x_cam: float, z_cam: float) -> tuple[float, float]:
    """
    Project a camera-space (x, z) point into BEV plane.
    Camera convention: z = forward, x = right.
    BEV convention: same — no rotation needed.
    """
    return x_cam, z_cam


def estimate_intrinsics(h: int, w: int) -> np.ndarray:
    """
    Estimate a pinhole intrinsics matrix from image dimensions.
    Uses focal length = max(h, w), principal point at image center.
    Matches WildDet3D's default when no calibration is provided.
    """
    f = float(max(h, w))
    K = np.array([
        [f,   0.0, w / 2.0],
        [0.0, f,   h / 2.0],
        [0.0, 0.0, 1.0    ],
    ], dtype=np.float32)
    return K


def world_to_grid(
    x_cam: float,
    z_cam: float,
    x_range: list[float],
    z_range: list[float],
    resolution: float,
) -> tuple[int, int] | None:
    """
    Convert camera-space (x, z) to BEV grid cell indices (row, col).
    Returns None if the point is outside the grid range.
    """
    if not (x_range[0] <= x_cam <= x_range[1]):
        return None
    if not (z_range[0] <= z_cam <= z_range[1]):
        return None

    col = int((x_cam - x_range[0]) / resolution)
    row = int((z_cam - z_range[0]) / resolution)
    return row, col
```

**Step 4: Run tests**

```bash
pytest tests/test_geometry.py -v
```

Expected: 4 PASS

**Step 5: Commit**

```bash
git add fusion_perception/utils/geometry.py tests/test_geometry.py
git commit -m "feat: add geometry utilities for BEV projection and intrinsics"
```

---

### Task 5: JSON I/O utilities

**Files:**
- Create: `fusion_perception/utils/json_io.py`
- Create: `tests/test_json_io.py`

**Step 1: Write failing test**

```python
# tests/test_json_io.py
import json
from pathlib import Path
import tempfile
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.json_io import save_detections, load_detections

def test_save_and_load_detections_roundtrip(tmp_path):
    det = Detection3D(
        frame_idx=0, class_id=0, class_name="car", score=0.9,
        score_2d=0.88, score_3d=0.92,
        box_2d=[10.0, 20.0, 100.0, 80.0],
        box_3d=[2.1, 0.8, 15.3, 1.8, 1.5, 4.2, 0.05],
        centroid_2d=[55.0, 50.0],
        centroid_3d=[2.1, 0.8, 15.3],
        depth=15.3,
    )
    out_path = tmp_path / "detections.json"
    save_detections(
        path=out_path,
        run_id="test_run",
        video_path="clip.mp4",
        prompts=["car"],
        frames_data={0: [det]},
    )
    assert out_path.exists()
    loaded = load_detections(out_path)
    assert loaded["run_id"] == "test_run"
    assert loaded["frames"]["0"][0]["class_name"] == "car"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_json_io.py -v
```

**Step 3: Implement `fusion_perception/utils/json_io.py`**

```python
"""Typed save/load for all pipeline intermediate JSON outputs."""
import json
from pathlib import Path
from typing import Any
from fusion_perception.utils.dataclasses import (
    Detection3D, Track, OccupancyGrid, SceneMemory, ReasoningOutput,
)


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _read(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_detections(
    path: Path,
    run_id: str,
    video_path: str,
    prompts: list[str],
    frames_data: dict[int, list[Detection3D]],
) -> None:
    payload = {
        "run_id": run_id,
        "video_path": str(video_path),
        "prompts": prompts,
        "frames": {
            str(idx): [d.to_dict() for d in dets]
            for idx, dets in frames_data.items()
        },
    }
    _write(path, payload)


def load_detections(path: Path) -> dict:
    return _read(path)


def save_tracks(
    path: Path,
    run_id: str,
    tracks: dict[int, Track],
) -> None:
    payload = {
        "run_id": run_id,
        "tracks": {str(tid): t.to_dict() for tid, t in tracks.items()},
    }
    _write(path, payload)


def load_tracks(path: Path) -> dict:
    return _read(path)


def save_occupancy(
    path: Path,
    run_id: str,
    config: dict,
    frames_data: dict[int, OccupancyGrid],
) -> None:
    payload = {
        "run_id": run_id,
        "config": config,
        "frames": {
            str(idx): {
                "frame_idx": g.frame_idx,
                "grid": g.grid,
                "occupied_cells": sum(
                    1 for row in g.grid for c in row if c > 0.5
                ),
                "free_cells": sum(
                    1 for row in g.grid for c in row if c <= 0.5
                ),
            }
            for idx, g in frames_data.items()
        },
    }
    _write(path, payload)


def save_reasoning(
    path: Path,
    run_id: str,
    reasoning_interval: int,
    outputs: list[ReasoningOutput],
) -> None:
    payload = {
        "run_id": run_id,
        "reasoning_interval": reasoning_interval,
        "outputs": [r.to_dict() for r in outputs],
    }
    _write(path, payload)


def load_reasoning(path: Path) -> dict:
    return _read(path)
```

**Step 4: Run tests**

```bash
pytest tests/test_json_io.py -v
```

Expected: 1 PASS

**Step 5: Commit**

```bash
git add fusion_perception/utils/json_io.py tests/test_json_io.py
git commit -m "feat: add typed JSON save/load for all pipeline stages"
```

---

### Task 6: Master YAML config

**Files:**
- Create: `configs/default.yaml`
- Create: `configs/detection.yaml`
- Create: `configs/tracking.yaml`
- Create: `configs/occupancy.yaml`
- Create: `configs/reasoning.yaml`
- Create: `configs/colab.yaml`
- Create: `tests/test_config.py`

**Step 1: Write failing test**

```python
# tests/test_config.py
from omegaconf import OmegaConf

def test_default_config_loads():
    cfg = OmegaConf.load("configs/default.yaml")
    assert "detection" in cfg
    assert "tracking" in cfg
    assert "occupancy" in cfg
    assert "reasoning" in cfg

def test_colab_overrides_merge():
    base = OmegaConf.load("configs/default.yaml")
    overrides = OmegaConf.load("configs/colab.yaml")
    merged = OmegaConf.merge(base, overrides)
    assert merged.detection.fp16 is True
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

**Step 3: Write all config files**

`configs/default.yaml`:
```yaml
run_id: null  # auto-generated from timestamp if null

video:
  source: "data/sample_videos/clip.mp4"
  max_frames: null  # null = process entire video
  resize_hw: [480, 640]  # [H, W] resize before detection

detection:
  checkpoint: "ckpt/wilddet3d_alldata_all_prompt_v1.0.pt"
  prompts: ["car", "person", "cyclist"]
  score_threshold: 0.4
  fp16: true
  device: "cuda"

tracking:
  window_size: 8              # CoWTracker frame buffer size
  max_tracks: 50
  occlusion_tolerance: 10     # frames before a track is killed
  nn_threshold: 50.0          # max pixel distance for centroid matching
  device: "cuda"

occupancy:
  resolution: 0.5             # meters per BEV cell
  x_range: [-20.0, 20.0]     # lateral range in meters
  z_range: [0.0, 50.0]       # forward range in meters
  decay_factor: 0.95          # per-frame exponential decay

reasoning:
  enabled: true
  interval_frames: 30         # run Gemma every N frames
  visual_mode: false          # true = send annotated frame to Gemma
  model_id: "google/gemma-4-2b-it"
  quantize_4bit: true
  max_new_tokens: 256
  device: "cuda"

output:
  base_dir: "outputs"
  flush_interval: 100         # write JSON every N frames
  save_video: true
  video_fps: null             # null = match input fps

logging:
  level: "INFO"
  log_file: null
  use_rich: true
```

`configs/colab.yaml`:
```yaml
# Colab-specific overrides — merge on top of default.yaml
detection:
  fp16: true
tracking:
  window_size: 4              # reduce if T4 memory pressure
reasoning:
  quantize_4bit: true
  visual_mode: false          # start text-only; flip to true manually
output:
  base_dir: "/content/drive/MyDrive/fusion_perception/outputs"
logging:
  level: "INFO"
  use_rich: false             # rich rendering can be noisy in Colab
```

`configs/detection.yaml`:
```yaml
checkpoint: "ckpt/wilddet3d_alldata_all_prompt_v1.0.pt"
prompts: ["car", "person", "cyclist"]
score_threshold: 0.4
fp16: true
device: "cuda"
```

`configs/tracking.yaml`:
```yaml
window_size: 8
max_tracks: 50
occlusion_tolerance: 10
nn_threshold: 50.0
device: "cuda"
```

`configs/occupancy.yaml`:
```yaml
resolution: 0.5
x_range: [-20.0, 20.0]
z_range: [0.0, 50.0]
decay_factor: 0.95
```

`configs/reasoning.yaml`:
```yaml
enabled: true
interval_frames: 30
visual_mode: false
model_id: "google/gemma-4-2b-it"
quantize_4bit: true
max_new_tokens: 256
device: "cuda"
```

**Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```

Expected: 2 PASS

**Step 5: Commit**

```bash
git add configs/ tests/test_config.py
git commit -m "feat: add YAML config system with Colab overrides"
```

---

## Phase 2 — Data Ingestion

### Task 7: Video loader

**Files:**
- Create: `fusion_perception/utils/video_loader.py`
- Create: `tests/test_video_loader.py`

**Step 1: Write failing test**

```python
# tests/test_video_loader.py
import numpy as np
from fusion_perception.utils.video_loader import VideoLoader

def test_video_loader_yields_frames_with_metadata(tmp_path):
    # create a tiny synthetic video using OpenCV
    import cv2
    vpath = str(tmp_path / "test.mp4")
    out = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        out.write(np.zeros((48, 64, 3), dtype=np.uint8))
    out.release()

    loader = VideoLoader(source=vpath, resize_hw=(48, 64), max_frames=None)
    frames = list(loader)
    assert len(frames) == 5
    idx, frame, meta = frames[0]
    assert idx == 0
    assert frame.shape == (48, 64, 3)
    assert "fps" in meta
    assert "total_frames" in meta

def test_video_loader_respects_max_frames(tmp_path):
    import cv2
    vpath = str(tmp_path / "test2.mp4")
    out = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(10):
        out.write(np.zeros((48, 64, 3), dtype=np.uint8))
    out.release()

    loader = VideoLoader(source=vpath, resize_hw=None, max_frames=3)
    frames = list(loader)
    assert len(frames) == 3
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_video_loader.py -v
```

**Step 3: Implement `fusion_perception/utils/video_loader.py`**

```python
"""Video frame iterator with metadata. Supports file paths and frame limits."""
from __future__ import annotations
import cv2
import numpy as np
from typing import Iterator, Optional
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("video_loader")


class VideoLoader:
    """Yields (frame_idx, frame_rgb, metadata) tuples from a video source."""

    def __init__(
        self,
        source: str,
        resize_hw: Optional[tuple[int, int]],
        max_frames: Optional[int],
    ) -> None:
        self.source = source
        self.resize_hw = resize_hw
        self.max_frames = max_frames
        self._cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.source}")
        return cap

    @property
    def fps(self) -> float:
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    @property
    def total_frames(self) -> int:
        cap = self._open()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, dict]]:
        cap = self._open()
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        meta = {"fps": fps, "total_frames": total, "original_hw": (h, w)}
        logger.info(f"Opened {self.source}: {w}x{h} @ {fps:.1f}fps, {total} frames")

        frame_idx = 0
        while True:
            if self.max_frames is not None and frame_idx >= self.max_frames:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if self.resize_hw is not None:
                th, tw = self.resize_hw
                frame_rgb = cv2.resize(frame_rgb, (tw, th))
            yield frame_idx, frame_rgb, meta
            frame_idx += 1

        cap.release()
        logger.info(f"VideoLoader finished: {frame_idx} frames yielded")
```

**Step 4: Run tests**

```bash
pytest tests/test_video_loader.py -v
```

Expected: 2 PASS

**Step 5: Commit**

```bash
git add fusion_perception/utils/video_loader.py tests/test_video_loader.py
git commit -m "feat: add VideoLoader with resize and max_frames support"
```

---

### Task 8: GPU memory monitor

**Files:**
- Create: `fusion_perception/utils/memory_monitor.py`

**Step 1: Implement `fusion_perception/utils/memory_monitor.py`**

```python
"""GPU and CPU memory monitoring. Logs warnings before OOM."""
from __future__ import annotations
from typing import Optional
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("memory_monitor")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def log_gpu_memory(tag: str = "") -> dict:
    """Log current GPU memory usage. Returns dict with stats."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved

    label = f"[{tag}] " if tag else ""
    logger.debug(
        f"{label}GPU: {allocated:.2f}GB alloc | "
        f"{reserved:.2f}GB reserved | "
        f"{free:.2f}GB free / {total:.2f}GB total"
    )

    if free < 1.0:
        logger.warning(f"{label}GPU memory low: {free:.2f}GB free — OOM risk")

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free,
        "total_gb": total,
    }


def clear_gpu_cache() -> None:
    """Free unused cached GPU memory."""
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")
```

**Step 2: Commit**

```bash
git add fusion_perception/utils/memory_monitor.py
git commit -m "feat: add GPU memory monitor with OOM warning"
```

---

## Phase 3 — Detection

### Task 9: Base detector abstract class

**Files:**
- Create: `fusion_perception/models/base_detector.py`
- Create: `fusion_perception/models/__init__.py`

**Step 1: Implement `fusion_perception/models/base_detector.py`**

```python
"""Abstract base class for 3D object detectors.

Implement this interface to swap WildDet3D for any future detector
without changing any downstream pipeline code.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D


class BaseDetector(ABC):

    @abstractmethod
    def load(self, checkpoint_path: str, device: str) -> None:
        """Load model weights onto device."""
        ...

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intrinsics: np.ndarray | None,
        prompts: list[str],
    ) -> list[Detection3D]:
        """
        Run detection on a single RGB frame [H,W,3] uint8.
        intrinsics: [3,3] float32, or None to estimate from frame size.
        Returns list of Detection3D sorted by score descending.
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Move model off GPU and free memory."""
        ...
```

**Step 2: Write `fusion_perception/models/__init__.py`**

```python
from .base_detector import BaseDetector
```

**Step 3: Commit**

```bash
git add fusion_perception/models/
git commit -m "feat: add BaseDetector abstract class"
```

---

### Task 10: WildDet3D wrapper

**Files:**
- Create: `fusion_perception/models/wilddet3d_wrapper.py`

**Step 1: Implement `fusion_perception/models/wilddet3d_wrapper.py`**

```python
"""WildDet3D inference wrapper.

Wraps the allenai/WildDet3D model behind the BaseDetector interface.
Handles FP16 casting, intrinsics estimation, and output parsing.

TODO: Add support for point and box prompts (currently text-only).
TODO: Expose depth map output for downstream occupancy fusion.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Optional
from fusion_perception.models.base_detector import BaseDetector
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.geometry import box2d_centroid, box3d_centroid, estimate_intrinsics
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("wilddet3d_wrapper")


class WildDet3DWrapper(BaseDetector):
    """
    Wraps WildDet3D for single-frame 3D detection.

    Input:  RGB frame [H,W,3] uint8 + text prompts
    Output: List[Detection3D] sorted by score descending
    """

    def __init__(self, score_threshold: float = 0.4, fp16: bool = True) -> None:
        self.score_threshold = score_threshold
        self.fp16 = fp16
        self.device = "cpu"
        self._model = None
        self._preprocess = None

    def load(self, checkpoint_path: str, device: str = "cuda") -> None:
        """Load WildDet3D model weights."""
        logger.info(f"Loading WildDet3D from {checkpoint_path} on {device}")
        try:
            from wilddet3d import build_model, preprocess as wpreprocess
        except ImportError:
            raise ImportError(
                "WildDet3D not installed. "
                "Run: pip install git+https://github.com/allenai/WildDet3D"
            )

        self.device = device
        dtype = torch.float16 if self.fp16 else torch.float32
        self._model = build_model(checkpoint_path).to(device).to(dtype)
        self._model.eval()
        self._preprocess = wpreprocess
        log_gpu_memory("WildDet3D loaded")
        logger.info("WildDet3D ready")

    @torch.no_grad()
    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intrinsics: Optional[np.ndarray],
        prompts: list[str],
    ) -> list[Detection3D]:
        """Run WildDet3D on a single frame."""
        if self._model is None:
            raise RuntimeError("Call load() before detect()")

        h, w = frame.shape[:2]
        if intrinsics is None:
            intrinsics = estimate_intrinsics(h, w)

        dtype = torch.float16 if self.fp16 else torch.float32
        data = self._preprocess(
            image=frame,
            intrinsics=intrinsics,
        )

        results = self._model(
            images=data["images"].to(self.device, dtype=dtype),
            intrinsics=data["intrinsics"].to(self.device, dtype=dtype)[None],
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            input_texts=prompts,
        )

        boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids, _ = results
        detections = []

        for i in range(len(scores[0])):
            score = float(scores[0][i])
            if score < self.score_threshold:
                continue

            b2d = boxes_2d[0][i].cpu().tolist()
            b3d = boxes_3d[0][i].cpu().tolist()
            cid = int(class_ids[0][i])

            cx2, cy2 = box2d_centroid(b2d)
            c3 = box3d_centroid(b3d)

            detections.append(Detection3D(
                frame_idx=frame_idx,
                class_id=cid,
                class_name=prompts[cid] if cid < len(prompts) else str(cid),
                score=score,
                score_2d=float(scores_2d[0][i]),
                score_3d=float(scores_3d[0][i]),
                box_2d=b2d,
                box_3d=b3d,
                centroid_2d=[cx2, cy2],
                centroid_3d=c3,
                depth=float(c3[2]),
            ))

        detections.sort(key=lambda d: d.score, reverse=True)
        logger.debug(f"Frame {frame_idx}: {len(detections)} detections")
        return detections

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("WildDet3D unloaded")
```

**Step 2: Commit**

```bash
git add fusion_perception/models/wilddet3d_wrapper.py
git commit -m "feat: add WildDet3D wrapper with FP16 and score filtering"
```

---

## Phase 4 — Tracking

### Task 11: Centroid anchor logic

**Files:**
- Create: `fusion_perception/tracking/centroid_anchor.py`
- Create: `tests/test_centroid_anchor.py`

**Step 1: Write failing test**

```python
# tests/test_centroid_anchor.py
from fusion_perception.tracking.centroid_anchor import (
    match_detections_to_tracks, assign_new_track_id
)
from fusion_perception.utils.dataclasses import Detection3D, Track

def _make_det(frame_idx, cx, cy, cz=15.0):
    return Detection3D(
        frame_idx=frame_idx, class_id=0, class_name="car",
        score=0.9, score_2d=0.88, score_3d=0.92,
        box_2d=[cx-20, cy-20, cx+20, cy+20],
        box_3d=[0.0, 0.0, cz, 1.8, 1.5, 4.2, 0.0],
        centroid_2d=[cx, cy], centroid_3d=[0.0, 0.0, cz], depth=cz,
    )

def _make_track(tid, cx, cy):
    return Track(
        track_id=tid, class_name="car",
        first_seen=0, last_seen=0,
        centroid_history=[[cx, cy]],
        position_3d_history=[[0.0, 0.0, 15.0]],
        cow_query_point=[cx, cy],
        is_active=True, occlusion_count=0,
    )

def test_nearby_detection_matches_existing_track():
    tracks = {1: _make_track(1, 100.0, 100.0)}
    dets = [_make_det(1, 102.0, 101.0)]   # 2.2px away — well within threshold
    matched, unmatched_dets, unmatched_tracks = match_detections_to_tracks(
        detections=dets, active_tracks=tracks, nn_threshold=50.0
    )
    assert (dets[0], 1) in matched
    assert len(unmatched_dets) == 0

def test_far_detection_creates_new_track():
    tracks = {1: _make_track(1, 100.0, 100.0)}
    dets = [_make_det(1, 500.0, 400.0)]   # far away
    matched, unmatched_dets, unmatched_tracks = match_detections_to_tracks(
        detections=dets, active_tracks=tracks, nn_threshold=50.0
    )
    assert len(matched) == 0
    assert len(unmatched_dets) == 1

def test_assign_new_track_id_increments():
    assert assign_new_track_id({}) == 1
    assert assign_new_track_id({1: None, 2: None}) == 3
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_centroid_anchor.py -v
```

**Step 3: Implement `fusion_perception/tracking/centroid_anchor.py`**

```python
"""Centroid-anchoring logic: match WildDet3D detections to CoWTracker query points.

Strategy: nearest-neighbour matching in 2D pixel space between detection
centroids and existing track query points. Hungarian assignment not used
here — NN is sufficient for the initial implementation.

TODO: Upgrade to Hungarian/IoU matching for crowded scenes.
"""
from __future__ import annotations
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D, Track


def _dist(a: list[float], b: list[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def match_detections_to_tracks(
    detections: list[Detection3D],
    active_tracks: dict[int, Track],
    nn_threshold: float,
) -> tuple[
    list[tuple[Detection3D, int]],   # matched: (detection, track_id)
    list[Detection3D],               # unmatched detections → new tracks
    list[int],                       # unmatched track IDs → occlusion
]:
    """
    Greedy nearest-neighbour matching between detection centroids and
    existing track query points.

    Processes detections in score order (highest first) to prioritise
    confident detections during assignment.
    """
    if not active_tracks:
        return [], list(detections), []

    matched: list[tuple[Detection3D, int]] = []
    unmatched_dets: list[Detection3D] = []
    used_track_ids: set[int] = set()

    for det in detections:
        best_tid = None
        best_dist = float("inf")
        for tid, track in active_tracks.items():
            if tid in used_track_ids:
                continue
            d = _dist(det.centroid_2d, track.cow_query_point)
            if d < best_dist:
                best_dist = d
                best_tid = tid

        if best_tid is not None and best_dist <= nn_threshold:
            matched.append((det, best_tid))
            used_track_ids.add(best_tid)
        else:
            unmatched_dets.append(det)

    unmatched_track_ids = [
        tid for tid in active_tracks if tid not in used_track_ids
    ]
    return matched, unmatched_dets, unmatched_track_ids


def assign_new_track_id(existing_tracks: dict) -> int:
    """Return the next available track ID (max existing + 1, or 1)."""
    if not existing_tracks:
        return 1
    return max(existing_tracks.keys()) + 1
```

**Step 4: Run tests**

```bash
pytest tests/test_centroid_anchor.py -v
```

Expected: 3 PASS

**Step 5: Commit**

```bash
git add fusion_perception/tracking/centroid_anchor.py tests/test_centroid_anchor.py
git commit -m "feat: add centroid anchor matching for detection-to-track assignment"
```

---

### Task 12: CoWTracker wrapper

**Files:**
- Create: `fusion_perception/tracking/cowtracker_wrapper.py`
- Create: `fusion_perception/tracking/base_tracker.py`
- Create: `fusion_perception/tracking/__init__.py`

**Step 1: Implement `fusion_perception/tracking/base_tracker.py`**

```python
"""Abstract base class for multi-object trackers."""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D, Track


class BaseTracker(ABC):

    @abstractmethod
    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection3D],
        frame_idx: int,
    ) -> list[Track]:
        """Update tracker with new frame and detections. Return active tracks."""
        ...

    @abstractmethod
    def get_all_tracks(self) -> dict[int, Track]:
        """Return full track registry including inactive tracks."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear all state between videos."""
        ...
```

**Step 2: Implement `fusion_perception/tracking/cowtracker_wrapper.py`**

```python
"""CoWTracker centroid-anchoring tracker.

Uses CoWTrackerWindowed to track the 2D centroid of each active object
across frames. WildDet3D detections are matched to existing tracks by
nearest-neighbour in pixel space (see centroid_anchor.py).

Flow per frame:
  1. Run CoWTracker on frame window → get updated positions for all queries
  2. Match new WildDet3D detections to updated track positions (NN)
  3. Spawn new tracks for unmatched detections
  4. Increment occlusion counter for unmatched tracks; kill if > tolerance

TODO: Tune window_size vs. memory tradeoff for longer videos.
TODO: Add re-ID via appearance features for severe occlusions.
"""
from __future__ import annotations
import numpy as np
import torch
from collections import deque
from typing import Optional
from fusion_perception.tracking.base_tracker import BaseTracker
from fusion_perception.tracking.centroid_anchor import (
    match_detections_to_tracks, assign_new_track_id
)
from fusion_perception.utils.dataclasses import Detection3D, Track
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("cowtracker_wrapper")


class CoWTrackerWrapper(BaseTracker):
    """
    Wraps CoWTrackerWindowed with centroid-anchoring for object tracking.

    CoWTracker tracks a set of 2D query points across a sliding window of
    frames. Here, each query point corresponds to a WildDet3D object centroid.
    """

    def __init__(
        self,
        window_size: int = 8,
        max_tracks: int = 50,
        occlusion_tolerance: int = 10,
        nn_threshold: float = 50.0,
        device: str = "cuda",
    ) -> None:
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.occlusion_tolerance = occlusion_tolerance
        self.nn_threshold = nn_threshold
        self.device = device

        self._model = None
        self._all_tracks: dict[int, Track] = {}
        self._active_ids: set[int] = set()
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)

    def load(self) -> None:
        """Download and load CoWTracker model weights."""
        logger.info("Loading CoWTrackerWindowed from HuggingFace Hub")
        try:
            from cotracker.models.build_cotracker import build_cotracker_from_cfg  # noqa
            from cotracker.predictor import CoTrackerPredictor
        except ImportError:
            raise ImportError(
                "CoWTracker not installed. "
                "Run: pip install git+https://github.com/facebookresearch/co-tracker"
            )
        self._model = CoTrackerPredictor(
            checkpoint=None,  # auto-download
            window_len=self.window_size,
        ).to(self.device)
        self._model.eval()
        log_gpu_memory("CoWTracker loaded")
        logger.info("CoWTracker ready")

    @torch.no_grad()
    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection3D],
        frame_idx: int,
    ) -> list[Track]:
        """
        Update tracker state with a new frame and its detections.
        Returns list of currently active tracks.
        """
        self._frame_buffer.append(frame)

        active_tracks = {
            tid: self._all_tracks[tid]
            for tid in self._active_ids
            if tid in self._all_tracks
        }

        # Step 1: run CoWTracker to update existing query point positions
        if self._model is not None and len(active_tracks) > 0:
            active_tracks = self._run_cowtracker(active_tracks, frame_idx)

        # Step 2: match new detections to updated track positions
        matched, unmatched_dets, unmatched_tids = match_detections_to_tracks(
            detections=detections,
            active_tracks=active_tracks,
            nn_threshold=self.nn_threshold,
        )

        # Step 3: update matched tracks
        for det, tid in matched:
            track = self._all_tracks[tid]
            track.last_seen = frame_idx
            track.centroid_history.append(det.centroid_2d)
            track.position_3d_history.append(det.centroid_3d)
            track.cow_query_point = det.centroid_2d
            track.occlusion_count = 0
            track.is_active = True

        # Step 4: handle unmatched tracks (occlusion)
        for tid in unmatched_tids:
            track = self._all_tracks[tid]
            track.occlusion_count += 1
            if track.occlusion_count > self.occlusion_tolerance:
                track.is_active = False
                self._active_ids.discard(tid)
                logger.debug(f"Track {tid} killed after {self.occlusion_tolerance} occlusion frames")

        # Step 5: spawn new tracks for unmatched detections
        for det in unmatched_dets:
            if len(self._active_ids) >= self.max_tracks:
                logger.warning(f"max_tracks={self.max_tracks} reached, skipping new track")
                break
            new_id = assign_new_track_id(self._all_tracks)
            new_track = Track(
                track_id=new_id,
                class_name=det.class_name,
                first_seen=frame_idx,
                last_seen=frame_idx,
                centroid_history=[det.centroid_2d],
                position_3d_history=[det.centroid_3d],
                cow_query_point=det.centroid_2d,
                is_active=True,
                occlusion_count=0,
            )
            self._all_tracks[new_id] = new_track
            self._active_ids.add(new_id)
            logger.debug(f"New track {new_id}: {det.class_name} @ {det.centroid_2d}")

        active_list = [
            self._all_tracks[tid]
            for tid in self._active_ids
            if tid in self._all_tracks
        ]
        logger.debug(f"Frame {frame_idx}: {len(active_list)} active tracks")
        return active_list

    def _run_cowtracker(
        self,
        active_tracks: dict[int, Track],
        frame_idx: int,
    ) -> dict[int, Track]:
        """
        Run CoWTracker on the current frame buffer and update query points.
        Returns updated active_tracks dict with new cow_query_point values.
        """
        if len(self._frame_buffer) < 2:
            return active_tracks

        frames_np = np.stack(list(self._frame_buffer), axis=0)  # [T,H,W,3]
        video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        video = video.unsqueeze(0).to(self.device)  # [1,T,3,H,W]

        track_ids = list(active_tracks.keys())
        queries = torch.tensor(
            [[0.0, t.cow_query_point[0], t.cow_query_point[1]]
             for t in active_tracks.values()],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)  # [1, N, 3]

        pred_tracks, pred_vis = self._model(video, queries=queries)
        # pred_tracks: [1, T, N, 2]

        last_pos = pred_tracks[0, -1]  # [N, 2] — positions in latest frame
        for i, tid in enumerate(track_ids):
            xy = last_pos[i].cpu().tolist()
            active_tracks[tid].cow_query_point = xy

        return active_tracks

    def get_all_tracks(self) -> dict[int, Track]:
        return self._all_tracks

    def reset(self) -> None:
        self._all_tracks = {}
        self._active_ids = set()
        self._frame_buffer.clear()
        logger.info("CoWTrackerWrapper reset")
```

**Step 3: Write `fusion_perception/tracking/__init__.py`**

```python
from .base_tracker import BaseTracker
from .cowtracker_wrapper import CoWTrackerWrapper
```

**Step 4: Commit**

```bash
git add fusion_perception/tracking/
git commit -m "feat: add CoWTracker centroid-anchoring tracker with occlusion handling"
```

---

## Phase 5 — Occupancy & Scene Memory

### Task 13: BEV occupancy grid

**Files:**
- Create: `fusion_perception/occupancy/bev_grid.py`
- Create: `fusion_perception/occupancy/__init__.py`
- Create: `tests/test_bev_grid.py`

**Step 1: Write failing test**

```python
# tests/test_bev_grid.py
from fusion_perception.occupancy.bev_grid import OccupancyBEVGenerator
from fusion_perception.utils.dataclasses import Track

def _make_track(tid, x, z):
    return Track(
        track_id=tid, class_name="car",
        first_seen=0, last_seen=1,
        centroid_history=[[100.0, 100.0]],
        position_3d_history=[[x, 0.0, z]],
        cow_query_point=[100.0, 100.0],
        is_active=True, occlusion_count=0,
    )

def test_occupied_cell_marked():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=1.0,  # no decay for clean test
    )
    tracks = [_make_track(1, x=0.0, z=10.0)]
    grid = gen.update(tracks, frame_idx=0)
    # x=0, z=10 → col=10, row=10
    assert grid.grid[10][10] == 1.0

def test_temporal_decay_reduces_occupancy():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=0.5,
    )
    tracks = [_make_track(1, x=0.0, z=10.0)]
    gen.update(tracks, frame_idx=0)
    grid2 = gen.update([], frame_idx=1)  # no objects — only decay
    assert grid2.grid[10][10] == 0.5
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_bev_grid.py -v
```

**Step 3: Implement `fusion_perception/occupancy/bev_grid.py`**

```python
"""BEV occupancy grid with temporal exponential decay.

Grid convention:
  rows  = forward (z) axis, row 0 = z_min
  cols  = lateral (x) axis, col 0 = x_min
  value = occupancy probability [0.0, 1.0]

TODO: Add ray-casting free-space estimation from ego origin.
TODO: Support multi-layer grids (height slices).
"""
from __future__ import annotations
import numpy as np
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
    ) -> None:
        self.resolution = resolution
        self.x_range = x_range
        self.z_range = z_range
        self.decay_factor = decay_factor

        n_rows = int((z_range[1] - z_range[0]) / resolution)
        n_cols = int((x_range[1] - x_range[0]) / resolution)
        self._grid = np.zeros((n_rows, n_cols), dtype=np.float32)

    def update(self, tracks: list[Track], frame_idx: int) -> OccupancyGrid:
        """Apply decay, rasterize current tracks, return updated grid."""
        self._grid *= self.decay_factor

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

**Step 4: Write `fusion_perception/occupancy/__init__.py`**

```python
from .bev_grid import OccupancyBEVGenerator
```

**Step 5: Run tests**

```bash
pytest tests/test_bev_grid.py -v
```

Expected: 2 PASS

**Step 6: Commit**

```bash
git add fusion_perception/occupancy/ tests/test_bev_grid.py
git commit -m "feat: add BEV occupancy grid with temporal decay"
```

---

### Task 14: Scene memory manager

**Files:**
- Create: `fusion_perception/tracking/trajectory_manager.py`

**Step 1: Implement `fusion_perception/tracking/trajectory_manager.py`**

```python
"""Scene memory: accumulates track + occupancy state, detects events.

Events currently detected:
  - new_object:<track_id>      track just spawned
  - lost_object:<track_id>     track just killed
  - sudden_stop:<track_id>     object velocity near zero after moving

TODO: Add sudden_appearance (object enters from frame edge).
TODO: Add trajectory_crossing (two track paths intersect).
"""
from __future__ import annotations
import numpy as np
from fusion_perception.utils.dataclasses import Track, OccupancyGrid, SceneMemory
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("trajectory_manager")


class SceneMemoryManager:
    """Aggregates perception state across frames. Detects semantic events."""

    def __init__(self, sudden_stop_threshold: float = 2.0) -> None:
        self.sudden_stop_threshold = sudden_stop_threshold
        self._prev_active_ids: set[int] = set()
        self._event_flags: list[str] = []
        self._frame_count: int = 0
        self._snapshot: SceneMemory | None = None

    def update(
        self,
        tracks: list[Track],
        occupancy: OccupancyGrid,
        frame_idx: int,
        fps: float,
    ) -> SceneMemory:
        """Update state, detect events, return current SceneMemory."""
        self._frame_count += 1
        current_ids = {t.track_id for t in tracks if t.is_active}

        # Detect new objects
        for tid in current_ids - self._prev_active_ids:
            self._event_flags.append(f"new_object:{tid}")
            logger.info(f"Event: new_object:{tid}")

        # Detect lost objects
        for tid in self._prev_active_ids - current_ids:
            self._event_flags.append(f"lost_object:{tid}")
            logger.info(f"Event: lost_object:{tid}")

        # Detect sudden stops
        for track in tracks:
            if len(track.centroid_history) >= 3:
                recent = np.array(track.centroid_history[-3:])
                velocities = np.linalg.norm(np.diff(recent, axis=0), axis=1)
                if velocities[-1] < self.sudden_stop_threshold and velocities[0] > self.sudden_stop_threshold * 3:
                    flag = f"sudden_stop:{track.track_id}"
                    if flag not in self._event_flags[-10:]:
                        self._event_flags.append(flag)

        self._prev_active_ids = current_ids
        elapsed = frame_idx / max(fps, 1.0)

        self._snapshot = SceneMemory(
            frame_idx=frame_idx,
            active_tracks=tracks,
            occupancy_grid=occupancy,
            event_flags=list(self._event_flags[-20:]),  # keep last 20 events
            frame_count=self._frame_count,
            elapsed_seconds=elapsed,
        )
        return self._snapshot

    def get_snapshot(self) -> SceneMemory | None:
        return self._snapshot

    def reset(self) -> None:
        self._prev_active_ids = set()
        self._event_flags = []
        self._frame_count = 0
        self._snapshot = None
```

**Step 2: Commit**

```bash
git add fusion_perception/tracking/trajectory_manager.py
git commit -m "feat: add SceneMemoryManager with event detection"
```

---

## Phase 6 — Reasoning & Visualization

### Task 15: Prompt builder

**Files:**
- Create: `fusion_perception/reasoning/prompt_builder.py`
- Create: `fusion_perception/reasoning/templates/scene_summary.txt`
- Create: `fusion_perception/reasoning/templates/anomaly_detect.txt`
- Create: `fusion_perception/reasoning/templates/trajectory_explain.txt`
- Create: `fusion_perception/reasoning/__init__.py`
- Create: `tests/test_prompt_builder.py`

**Step 1: Write failing test**

```python
# tests/test_prompt_builder.py
from fusion_perception.reasoning.prompt_builder import build_scene_prompt
from fusion_perception.utils.dataclasses import (
    Track, OccupancyGrid, SceneMemory
)

def _make_memory():
    track = Track(
        track_id=1, class_name="car", first_seen=0, last_seen=10,
        centroid_history=[[100.0, 100.0], [110.0, 98.0]],
        position_3d_history=[[2.0, 0.0, 15.0], [2.1, 0.0, 14.0]],
        cow_query_point=[110.0, 98.0],
        is_active=True, occlusion_count=0,
    )
    grid = OccupancyGrid(
        frame_idx=10, resolution=0.5,
        x_range=[-20.0, 20.0], z_range=[0.0, 50.0],
        grid=[[0.0] * 80 for _ in range(100)],
        decay_factor=0.95,
    )
    return SceneMemory(
        frame_idx=10, active_tracks=[track], occupancy_grid=grid,
        event_flags=["new_object:1"], frame_count=10, elapsed_seconds=0.33,
    )

def test_prompt_contains_track_info():
    memory = _make_memory()
    prompt = build_scene_prompt(memory)
    assert "car" in prompt
    assert "track_id=1" in prompt or "Track 1" in prompt

def test_prompt_contains_event_flags():
    memory = _make_memory()
    prompt = build_scene_prompt(memory)
    assert "new_object" in prompt
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_prompt_builder.py -v
```

**Step 3: Write prompt templates**

`fusion_perception/reasoning/templates/scene_summary.txt`:
```
You are an autonomous driving perception system analyzing a traffic scene.

Current frame: {frame_idx} | Elapsed: {elapsed:.1f}s | Active objects: {n_tracks}

=== TRACKED OBJECTS ===
{track_descriptions}

=== BEV OCCUPANCY ===
Occupied cells: {occupied_cells} / {total_cells} ({occupancy_pct:.1f}%)

=== EVENTS ===
{event_flags}

Provide a concise scene summary (2-3 sentences) covering:
1. What objects are present and their motion
2. Any unusual events or risks
3. Overall scene assessment
```

`fusion_perception/reasoning/templates/anomaly_detect.txt`:
```
Review the following perception data and identify any anomalies or safety-relevant events.

Events detected: {event_flags}
Active tracks: {n_tracks}
Track details: {track_descriptions}

List any anomalies as a bullet list. If none, say "No anomalies detected."
```

`fusion_perception/reasoning/templates/trajectory_explain.txt`:
```
Describe the motion of each tracked object in plain English.

{track_descriptions}

For each object, describe: direction of travel, approximate speed, and predicted near-term path.
Keep each description to one sentence.
```

**Step 4: Implement `fusion_perception/reasoning/prompt_builder.py`**

```python
"""Converts SceneMemory into structured Gemma prompts.

Uses txt templates from reasoning/templates/.
TODO: Add velocity estimation from centroid history deltas.
TODO: Support custom template injection via config.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from fusion_perception.utils.dataclasses import SceneMemory, Track

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _load_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text()


def _describe_track(track: Track, fps: float = 30.0) -> str:
    """Convert a Track to a single-line natural language description."""
    n_hist = len(track.centroid_history)
    if n_hist >= 2:
        delta = np.array(track.centroid_history[-1]) - np.array(track.centroid_history[-2])
        speed_px = float(np.linalg.norm(delta))
    else:
        speed_px = 0.0

    depth = track.position_3d_history[-1][2] if track.position_3d_history else 0.0
    return (
        f"Track {track.track_id} ({track.class_name}): "
        f"depth={depth:.1f}m, speed={speed_px:.1f}px/frame, "
        f"seen {track.last_seen - track.first_seen + 1} frames"
    )


def build_scene_prompt(memory: SceneMemory, fps: float = 30.0) -> str:
    """Build the primary scene summary prompt from SceneMemory."""
    template = _load_template("scene_summary.txt")

    track_descriptions = "\n".join(
        _describe_track(t, fps) for t in memory.active_tracks
    ) or "No active tracks."

    grid = memory.occupancy_grid
    total_cells = sum(len(row) for row in grid.grid)
    occupied_cells = sum(1 for row in grid.grid for c in row if c > 0.5)
    occupancy_pct = 100 * occupied_cells / max(total_cells, 1)

    event_str = "\n".join(memory.event_flags) if memory.event_flags else "None"

    return template.format(
        frame_idx=memory.frame_idx,
        elapsed=memory.elapsed_seconds,
        n_tracks=len(memory.active_tracks),
        track_descriptions=track_descriptions,
        occupied_cells=occupied_cells,
        total_cells=total_cells,
        occupancy_pct=occupancy_pct,
        event_flags=event_str,
    )
```

**Step 5: Write `fusion_perception/reasoning/__init__.py`**

```python
from .prompt_builder import build_scene_prompt
```

**Step 6: Run tests**

```bash
pytest tests/test_prompt_builder.py -v
```

Expected: 2 PASS

**Step 7: Commit**

```bash
git add fusion_perception/reasoning/ tests/test_prompt_builder.py
git commit -m "feat: add prompt builder with scene summary templates"
```

---

### Task 16: Gemma reasoning wrapper

**Files:**
- Create: `fusion_perception/models/gemma_wrapper.py`

**Step 1: Implement `fusion_perception/models/gemma_wrapper.py`**

```python
"""Gemma 4 2B reasoning wrapper.

Supports text-only and visual (frame + text) modes via config flag.
Uses bitsandbytes INT4 quantization by default to fit on T4.

Visual mode: annotated frame is passed as PIL image alongside text prompt.
Text mode: prompt only — faster, lower memory, suitable for high-frequency use.

TODO: Add streaming token generation for long summaries.
TODO: Cache KV for repeated prompt prefixes (same system prompt).
"""
from __future__ import annotations
import time
import numpy as np
from PIL import Image
from typing import Optional
from fusion_perception.utils.dataclasses import SceneMemory, ReasoningOutput
from fusion_perception.reasoning.prompt_builder import build_scene_prompt
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("gemma_wrapper")


class GemmaReasoningWrapper:
    """
    Wraps Gemma 4 2B-IT for semantic scene reasoning.
    Call load() once, then reason() per reasoning trigger.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-2b-it",
        quantize_4bit: bool = True,
        max_new_tokens: int = 256,
        device: str = "cuda",
        visual_mode: bool = False,
    ) -> None:
        self.model_id = model_id
        self.quantize_4bit = quantize_4bit
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.visual_mode = visual_mode
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load Gemma model with optional INT4 quantization."""
        logger.info(f"Loading {self.model_id} (4bit={self.quantize_4bit})")
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        kwargs: dict = {"device_map": self.device}

        if self.quantize_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        log_gpu_memory("Gemma loaded")
        logger.info("Gemma ready")

    def reason(
        self,
        memory: SceneMemory,
        annotated_frame: Optional[np.ndarray],
        trigger_reason: str = "interval",
        fps: float = 30.0,
    ) -> ReasoningOutput:
        """
        Run Gemma on current scene memory.
        annotated_frame: RGB [H,W,3] uint8, or None for text-only mode.
        """
        if self._model is None:
            raise RuntimeError("Call load() before reason()")

        use_visual = self.visual_mode and annotated_frame is not None
        prompt_text = build_scene_prompt(memory, fps=fps)

        t0 = time.perf_counter()
        if use_visual:
            raw = self._run_visual(prompt_text, annotated_frame)
        else:
            raw = self._run_text(prompt_text)
        latency_ms = (time.perf_counter() - t0) * 1000

        summary, anomalies, trajectory_nl = self._parse_response(raw)

        logger.info(
            f"Frame {memory.frame_idx} reasoning: {latency_ms:.0f}ms "
            f"({'visual' if use_visual else 'text'})"
        )

        return ReasoningOutput(
            frame_idx=memory.frame_idx,
            trigger_reason=trigger_reason,
            visual_mode=use_visual,
            prompt_used=prompt_text,
            summary=summary,
            anomalies=anomalies,
            trajectory_nl=trajectory_nl,
            raw_response=raw,
            latency_ms=latency_ms,
        )

    def _run_text(self, prompt: str) -> str:
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        return self._processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _run_visual(self, prompt: str, frame: np.ndarray) -> str:
        import torch
        pil_image = Image.fromarray(frame)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        return self._processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _parse_response(self, raw: str) -> tuple[str, list[str], str]:
        """Extract summary, anomalies, and trajectory description from raw Gemma output."""
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        summary = lines[0] if lines else raw
        anomalies = [l.lstrip("•-* ") for l in lines if l.startswith(("•", "-", "*"))]
        trajectory_nl = " ".join(lines[1:]) if len(lines) > 1 else ""
        return summary, anomalies, trajectory_nl

    def unload(self) -> None:
        import torch
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("Gemma unloaded")
```

**Step 2: Commit**

```bash
git add fusion_perception/models/gemma_wrapper.py
git commit -m "feat: add Gemma 4 2B wrapper with INT4 quantization and visual mode"
```

---

### Task 17: Visualization engine

**Files:**
- Create: `fusion_perception/visualization/frame_annotator.py`
- Create: `fusion_perception/visualization/bev_renderer.py`
- Create: `fusion_perception/visualization/output_compositor.py`
- Create: `fusion_perception/visualization/__init__.py`

**Step 1: Implement `fusion_perception/visualization/frame_annotator.py`**

```python
"""Draw 2D boxes, track IDs, scores, and trajectory tails on video frames."""
from __future__ import annotations
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D, Track

_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def annotate_frame(
    frame: np.ndarray,
    tracks: list[Track],
    detections: list[Detection3D],
    reasoning_text: str = "",
) -> np.ndarray:
    """Return annotated copy of frame with boxes, IDs, and trajectories."""
    out = frame.copy()

    # Draw trajectory tails
    for track in tracks:
        color = _color(track.track_id)
        hist = track.centroid_history[-20:]  # last 20 positions
        for i in range(1, len(hist)):
            p1 = (int(hist[i-1][0]), int(hist[i-1][1]))
            p2 = (int(hist[i][0]), int(hist[i][1]))
            alpha = i / len(hist)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(out, p1, p2, faded, 2)

    # Draw detection boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box_2d]
        tid = next((t.track_id for t in tracks if t.class_name == det.class_name
                    and abs(t.cow_query_point[0] - det.centroid_2d[0]) < 30), -1)
        color = _color(tid) if tid >= 0 else (200, 200, 200)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {det.class_name} {det.score:.2f} d={det.depth:.1f}m"
        cv2.putText(out, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Overlay reasoning text
    if reasoning_text:
        for i, line in enumerate(reasoning_text.split(". ")[:3]):
            cv2.putText(out, line.strip(), (8, out.shape[0] - 12 - i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return out
```

**Step 2: Implement `fusion_perception/visualization/bev_renderer.py`**

```python
"""Render BEV occupancy grid as a top-down image with track positions."""
from __future__ import annotations
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import OccupancyGrid, Track
from fusion_perception.utils.geometry import world_to_grid


def render_bev(
    grid: OccupancyGrid,
    tracks: list[Track],
    size_px: int = 400,
) -> np.ndarray:
    """Render occupancy grid and track positions as a BEV image [size_px, size_px, 3]."""
    rows = len(grid.grid)
    cols = len(grid.grid[0]) if rows > 0 else 1

    occupancy = np.array(grid.grid, dtype=np.float32)
    bev_img = np.zeros((rows, cols, 3), dtype=np.uint8)

    # Map occupancy to red channel
    bev_img[:, :, 2] = (occupancy * 255).astype(np.uint8)
    # Free space as dark green
    bev_img[:, :, 1] = ((1.0 - occupancy) * 40).astype(np.uint8)

    # Draw track centroids as colored dots
    for track in tracks:
        if not track.position_3d_history:
            continue
        x, _, z = track.position_3d_history[-1]
        cell = world_to_grid(x, z, grid.x_range, grid.z_range, grid.resolution)
        if cell:
            row, col = cell
            color = (255, 200, 0)
            cv2.circle(bev_img, (col, row), 3, color, -1)

            # Draw motion arrow from previous position
            if len(track.position_3d_history) >= 2:
                px, _, pz = track.position_3d_history[-2]
                prev_cell = world_to_grid(px, pz, grid.x_range, grid.z_range, grid.resolution)
                if prev_cell:
                    pr, pc = prev_cell
                    cv2.arrowedLine(bev_img, (pc, pr), (col, row), (0, 255, 255), 1, tipLength=0.4)

    # Flip vertically: row 0 = z_min (near), want near at bottom
    bev_img = cv2.flip(bev_img, 0)
    # Add ego marker at bottom center
    ego_col = cols // 2
    cv2.circle(bev_img, (ego_col, rows - 2), 4, (0, 255, 0), -1)

    return cv2.resize(bev_img, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
```

**Step 3: Implement `fusion_perception/visualization/output_compositor.py`**

```python
"""Tile annotated frame + BEV + text panel into a single output image."""
from __future__ import annotations
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import ReasoningOutput


def composite(
    annotated_frame: np.ndarray,
    bev_image: np.ndarray,
    reasoning: ReasoningOutput | None,
    target_width: int = 1280,
) -> np.ndarray:
    """
    Produce side-by-side composite: [annotated_frame | bev | text_panel].
    All panels resized to the same height.
    """
    h = annotated_frame.shape[0]

    # Resize BEV to match frame height
    bev_h = cv2.resize(bev_image, (h, h))

    # Build text panel
    panel_w = target_width - annotated_frame.shape[1] - h
    panel_w = max(panel_w, 200)
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

    if reasoning:
        lines = [
            "=== SCENE REASONING ===",
            "",
            *reasoning.summary.split(". "),
            "",
            "Anomalies:",
            *[f"  - {a}" for a in reasoning.anomalies],
            "",
            f"Latency: {reasoning.latency_ms:.0f}ms",
            f"Mode: {'visual' if reasoning.visual_mode else 'text'}",
        ]
        for i, line in enumerate(lines[:30]):
            y = 16 + i * 16
            if y > h - 8:
                break
            cv2.putText(panel, line, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    return np.concatenate([annotated_frame, bev_h, panel], axis=1)
```

**Step 4: Write `fusion_perception/visualization/__init__.py`**

```python
from .frame_annotator import annotate_frame
from .bev_renderer import render_bev
from .output_compositor import composite
```

**Step 5: Commit**

```bash
git add fusion_perception/visualization/
git commit -m "feat: add frame annotator, BEV renderer, and output compositor"
```

---

## Phase 7 — Pipeline Integration

### Task 18: Stage runner and streaming pipeline

**Files:**
- Create: `fusion_perception/pipelines/streaming_pipeline.py`
- Create: `fusion_perception/pipelines/stage_runner.py`
- Create: `fusion_perception/pipelines/__init__.py`

**Step 1: Implement `fusion_perception/pipelines/stage_runner.py`**

```python
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
    ) -> dict:
        """
        Process one frame through all stages.
        Returns dict with all stage outputs and composite visualization.
        """
        detections: list[Detection3D] = self.detector.detect(
            frame, frame_idx, intrinsics, self.prompts
        )
        tracks: list[Track] = self.tracker.update(frame, detections, frame_idx)
        occupancy: OccupancyGrid = self.bev_generator.update(tracks, frame_idx)
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

**Step 2: Implement `fusion_perception/pipelines/streaming_pipeline.py`**

```python
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

        # Accumulated outputs for JSON flush
        self._all_detections: dict = {}
        self._all_tracks: dict = {}
        self._all_occupancy: dict = {}
        self._all_reasoning: list = []

        self._video_writer = None

    def _init_models(self) -> StageRunner:
        """Load all models onto GPU."""
        cfg = self.cfg
        logger.info("Initializing models...")

        detector = WildDet3DWrapper(
            score_threshold=cfg.detection.score_threshold,
            fp16=cfg.detection.fp16,
        )
        detector.load(cfg.detection.checkpoint, cfg.detection.device)

        tracker = CoWTrackerWrapper(
            window_size=cfg.tracking.window_size,
            max_tracks=cfg.tracking.max_tracks,
            occlusion_tolerance=cfg.tracking.occlusion_tolerance,
            nn_threshold=cfg.tracking.nn_threshold,
            device=cfg.tracking.device,
        )
        tracker.load()

        bev = OccupancyBEVGenerator(
            resolution=cfg.occupancy.resolution,
            x_range=list(cfg.occupancy.x_range),
            z_range=list(cfg.occupancy.z_range),
            decay_factor=cfg.occupancy.decay_factor,
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
            fps=30.0,  # updated after first frame
            visual_reasoning=cfg.reasoning.visual_mode,
        )

    def run(self, video_path: str) -> None:
        """Main frame loop."""
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Processing: {video_path}")

        runner = self._init_models()
        loader = VideoLoader(
            source=video_path,
            resize_hw=list(self.cfg.video.resize_hw),
            max_frames=self.cfg.video.max_frames,
        )

        if self.cfg.output.save_video:
            self._init_video_writer(loader)

        flush_interval = self.cfg.output.flush_interval

        for frame_idx, frame, meta in loader:
            runner.fps = meta["fps"]

            outputs = runner.run_frame(frame, frame_idx)

            # Accumulate outputs
            self._all_detections[frame_idx] = outputs["detections"]
            for t in outputs["tracks"]:
                self._all_tracks[t.track_id] = t
            self._all_occupancy[frame_idx] = outputs["occupancy"]
            if outputs["reasoning"]:
                self._all_reasoning.append(outputs["reasoning"])

            # Write video frame
            if self._video_writer is not None:
                bgr = outputs["composite"][:, :, ::-1]
                self._video_writer.write(bgr)

            # Periodic JSON flush
            if (frame_idx + 1) % flush_interval == 0:
                self._flush_outputs()
                logger.info(f"Flushed outputs at frame {frame_idx}")

        self._flush_outputs()
        if self._video_writer:
            self._video_writer.release()
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
            str(self.cfg.video.source), list(self.cfg.detection.prompts),
            self._all_detections,
        )
        save_tracks(base / "tracks.json", self.run_id, self._all_tracks)
        save_occupancy(base / "occupancy.json", self.run_id, cfg_dict, self._all_occupancy)
        save_reasoning(
            base / "reasoning.json", self.run_id,
            self.cfg.reasoning.interval_frames, self._all_reasoning,
        )
```

**Step 3: Write `fusion_perception/pipelines/__init__.py`**

```python
from .streaming_pipeline import StreamingPipeline
```

**Step 4: Commit**

```bash
git add fusion_perception/pipelines/
git commit -m "feat: add streaming pipeline with stage runner and JSON flush"
```

---

### Task 19: CLI entrypoint

**Files:**
- Create: `run.py`

**Step 1: Implement `run.py`**

```python
"""CLI entrypoint for the fusion perception pipeline.

Usage:
    python run.py --config configs/default.yaml
    python run.py --config configs/default.yaml --video path/to/clip.mp4
    python run.py --config configs/default.yaml configs/colab.yaml  # merge configs
"""
import argparse
from omegaconf import OmegaConf
from fusion_perception.pipelines.streaming_pipeline import StreamingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fusion Perception Pipeline")
    parser.add_argument("--config", nargs="+", default=["configs/default.yaml"],
                        help="One or more YAML config files (merged left-to-right)")
    parser.add_argument("--video", default=None,
                        help="Override video source path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in args.config])
    if args.video:
        cfg.video.source = args.video

    pipeline = StreamingPipeline(cfg)
    pipeline.run(cfg.video.source)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add run.py
git commit -m "feat: add CLI entrypoint with multi-config merge support"
```

---

### Task 20: Colab setup notebook

**Files:**
- Create: `notebooks/00_setup.ipynb`

**Step 1:** Write a Colab notebook that:
1. Clones the repo and installs deps
2. Downloads WildDet3D checkpoint from HuggingFace
3. Verifies GPU is available and logs VRAM
4. Runs a smoke test on a single random frame

Content outline (implement as notebook cells):

```python
# Cell 1: Clone and install
!git clone https://github.com/YOUR_USERNAME/fusion_perception
%cd fusion_perception
!pip install -e .
!pip install git+https://github.com/allenai/WildDet3D
!pip install git+https://github.com/facebookresearch/co-tracker

# Cell 2: Download checkpoint
!huggingface-cli download allenai/WildDet3D \
  wilddet3d_alldata_all_prompt_v1.0.pt --local-dir ckpt/

# Cell 3: Verify GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 4: Smoke test
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.geometry import estimate_intrinsics
import numpy as np
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
K = estimate_intrinsics(480, 640)
print("Smoke test passed:", K.shape)
```

**Step 2: Commit**

```bash
git add notebooks/
git commit -m "feat: add Colab setup notebook with smoke test"
```

---

## Summary

| Phase | Tasks | Key Output |
|-------|-------|-----------|
| 1 — Foundation | 1–6 | Dataclasses, logging, geometry, JSON I/O, YAML config |
| 2 — Data Ingestion | 7–8 | VideoLoader, GPU monitor |
| 3 — Detection | 9–10 | BaseDetector, WildDet3DWrapper |
| 4 — Tracking | 11–12 | CentroidAnchor, CoWTrackerWrapper |
| 5 — Occupancy | 13–14 | BEV grid, SceneMemoryManager |
| 6 — Reasoning & Viz | 15–17 | PromptBuilder, GemmaWrapper, Visualization |
| 7 — Integration | 18–20 | StreamingPipeline, CLI, Colab notebook |

Run all tests at any point with:
```bash
pytest tests/ -v
```
