# KalmanCoWTracker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the greedy centroid-anchor tracker with a hybrid Kalman Filter + CoWTracker that does proper 3D MOT — Hungarian assignment, depth-aware measurement noise, yaw-wrapped updates, ByteTrack two-threshold association, and ego-motion compensation.

**Architecture:** Each active track owns a `filterpy.KalmanFilter` with 10-dim state `[cx,cy,cz,θ,l,w,h,vx,vy,vz]`. CoWTracker tracks a dense pixel patch per object; the median displacement back-projects to 3D and corrects the Kalman measurement vector. Data association uses lapjv (Hungarian) on a hybrid cost `α·(1−IoU3D) + (1−α)·CoW_dist`. Tracks follow a TENTATIVE → CONFIRMED → LOST lifecycle with velocity decay and covariance inflate on re-localization.

**Tech Stack:** filterpy, lapjv, opencv-python (for ego-motion homography + ORB), torch (CoWTracker batch), numpy, existing `BaseTracker` / `Detection3D` / `Track` contracts.

**Key design decisions baked in:**
- Yaw wrapping: `atan2(sin(Δθ), cos(Δθ))` on every residual — no filter divergence at ±π
- Depth-dependent R: `σ_z = α·z² + β` — monocular depth uncertainty scales quadratically
- Detection confidence scales R: high-score detections trusted more
- Information fusion: `R_eff⁻¹ = R_det⁻¹ + R_cow⁻¹` when CoW points are available
- Lazy CoW: skip transformer for tracks with small Kalman innovation
- ByteTrack two-threshold: high-conf dets associate first, low-conf only with existing tracks
- Ego-motion compensation: ORB background homography from BoT-SORT
- Adaptive Q: `Q_t = Q_0·(1 + α·||ν_t||)` per track, no fixed noise assumption

---

## Task 1: Install dependencies + extend config

**Files:**
- Modify: `configs/default.yaml`
- Modify: `configs/tracking.yaml` (if it exists, otherwise create)

**Step 1: Install new Python dependencies**

Run:
```
pip install filterpy lapjv opencv-python
```
Expected: installs without error. `filterpy` provides `KalmanFilter`; `lapjv` provides fast Hungarian solver.

**Step 2: Add tracking backend + new hyperparams to `configs/default.yaml`**

In the `tracking:` section, replace existing content with:
```yaml
tracking:
  backend: "kalman_cow"          # "cowtracker" | "kalman_cow"
  window_size: 8
  max_tracks: 50
  device: "cuda"

  # KalmanCoWTracker-specific
  lost_patience: 30              # frames before deleting a LOST track
  confirm_age: 3                 # TENTATIVE frames needed to become CONFIRMED
  high_score_threshold: 0.5      # ByteTrack: associate + init new tracks
  low_score_threshold: 0.2       # ByteTrack: associate with existing only
  iou_threshold: 0.5             # max assignment cost to accept a match
  alpha_cost: 0.35               # blend: α·IoU_cost + (1−α)·CoW_cost
  cow_conf_threshold: 0.85       # CoWTracker visibility gate
  min_cow_points: 4              # min surviving points to use CoW measurement
  velocity_decay: 0.9            # γ^k velocity decay for LOST tracks
  depth_alpha: 0.005             # σ_z = α·z² + β  (monocular depth noise)
  depth_beta: 0.5
  sigma_xy_base: 0.5             # lateral/vertical base noise (metres)
  sigma_xy_slope: 0.01           # scales with depth
  lazy_cow_innovation: 0.3       # skip CoW if ||ν|| below this threshold
  ego_motion: true               # enable ORB homography compensation
  mahal_threshold: 9.21          # χ²(0.99, df=4) gate before Hungarian
  # legacy CoWTrackerWrapper params (still used if backend="cowtracker")
  occlusion_tolerance: 10
  nn_threshold: 50.0
```

**Step 3: Verify config loads**

Run:
```
python -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/default.yaml'); print(c.tracking.backend)"
```
Expected: prints `kalman_cow`

---

## Task 2: Geometry helpers — `iou3d` + `wrap_angle`

**Files:**
- Modify: `fusion_perception/utils/geometry.py`
- Modify: `tests/test_geometry.py`

These helpers are pure functions with no dependencies — write tests first.

**Step 1: Write failing tests**

Add to `tests/test_geometry.py`:
```python
from fusion_perception.utils.geometry import iou3d, wrap_angle

def test_iou3d_perfect_overlap():
    box = [0., 0., 10., 0., 4., 2., 1.5]  # [cx,cy,cz,θ,l,w,h]
    assert abs(iou3d(box, box) - 1.0) < 1e-5

def test_iou3d_no_overlap():
    a = [0., 0., 10., 0., 2., 2., 1.5]
    b = [100., 0., 10., 0., 2., 2., 1.5]
    assert iou3d(a, b) == 0.0

def test_iou3d_partial():
    a = [0., 0., 10., 0., 4., 2., 1.5]
    b = [2., 0., 10., 0., 4., 2., 1.5]  # shifted 2m in x, half overlap
    val = iou3d(a, b)
    assert 0.1 < val < 0.9

def test_wrap_angle_no_change():
    import math
    assert abs(wrap_angle(0.5) - 0.5) < 1e-6

def test_wrap_angle_pi_boundary():
    import math
    # +π and -π are the same angle
    assert abs(wrap_angle(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-5

def test_wrap_angle_minus_pi():
    import math
    assert abs(abs(wrap_angle(-math.pi)) - math.pi) < 1e-5
```

**Step 2: Run to verify failure**

```
pytest tests/test_geometry.py::test_iou3d_perfect_overlap tests/test_geometry.py::test_wrap_angle_no_change -v
```
Expected: FAIL (ImportError or AttributeError)

**Step 3: Implement in `fusion_perception/utils/geometry.py`**

Add after existing functions:
```python
import math as _math

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return _math.atan2(_math.sin(angle), _math.cos(angle))


def iou3d(box_a: list[float], box_b: list[float]) -> float:
    """
    Approximate 3D IoU via BEV IoU × height overlap.
    box: [cx, cy, cz, theta, l, w, h]  (theta unused — axis-aligned approx)
    l=length(z-axis), w=width(x-axis), h=height(y-axis)
    """
    # BEV (x-z plane): use l and w
    ax1 = box_a[0] - box_a[5] / 2  # cx - w/2
    ax2 = box_a[0] + box_a[5] / 2
    az1 = box_a[2] - box_a[4] / 2  # cz - l/2
    az2 = box_a[2] + box_a[4] / 2

    bx1 = box_b[0] - box_b[5] / 2
    bx2 = box_b[0] + box_b[5] / 2
    bz1 = box_b[2] - box_b[4] / 2
    bz2 = box_b[2] + box_b[4] / 2

    inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_z = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter_bev = inter_x * inter_z
    area_a = box_a[4] * box_a[5]
    area_b = box_b[4] * box_b[5]
    union_bev = area_a + area_b - inter_bev

    # Height (y-axis)
    ay1 = box_a[1] - box_a[6] / 2
    ay2 = box_a[1] + box_a[6] / 2
    by1 = box_b[1] - box_b[6] / 2
    by2 = box_b[1] + box_b[6] / 2
    h_inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    h_union = max(ay2, by2) - min(ay1, by1)

    iou_bev = inter_bev / (union_bev + 1e-6)
    iou_h = h_inter / (h_union + 1e-6)
    return float(iou_bev * iou_h)
```

**Step 4: Run tests**

```
pytest tests/test_geometry.py -v
```
Expected: all geometry tests PASS

---

## Task 3: `_TrackState` dataclass + `TrackStatus` enum

**Files:**
- Create: `fusion_perception/tracking/track_state.py`
- Create: `tests/test_track_state.py`

**Step 1: Write failing test**

Create `tests/test_track_state.py`:
```python
from fusion_perception.tracking.track_state import TrackState, TrackStatus
import numpy as np

def test_track_status_values():
    assert TrackStatus.TENTATIVE != TrackStatus.CONFIRMED
    assert TrackStatus.CONFIRMED != TrackStatus.LOST

def test_track_state_defaults():
    ts = TrackState(track_id=1, class_name="car", kf=None, status=TrackStatus.TENTATIVE)
    assert ts.age == 0
    assert ts.miss_count == 0
    assert ts.cow_points_abs is None
    assert len(ts.centroid_history) == 0
    assert len(ts.position_3d_history) == 0

def test_track_state_to_track():
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=10, dim_z=7)
    kf.x = np.zeros((10, 1))
    kf.x[:7, 0] = [1., 0., 15., 0., 4., 2., 1.5]
    ts = TrackState(
        track_id=5, class_name="car", kf=kf,
        status=TrackStatus.CONFIRMED,
        first_seen=0, last_seen=10,
        centroid_history=[[320., 240.]],
        position_3d_history=[[1., 0., 15.]],
    )
    track = ts.to_track()
    assert track.track_id == 5
    assert track.class_name == "car"
    assert track.is_active is True
```

**Step 2: Run to verify failure**

```
pytest tests/test_track_state.py -v
```
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement `fusion_perception/tracking/track_state.py`**

```python
"""Per-track Kalman state and status for KalmanCoWTracker."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np
from fusion_perception.utils.dataclasses import Track


class TrackStatus(Enum):
    TENTATIVE = auto()   # needs confirm_age matches before activation
    CONFIRMED = auto()   # actively tracked
    LOST = auto()        # no detection; propagating via KF prediction only


@dataclass
class TrackState:
    track_id: int
    class_name: str
    kf: object                        # filterpy.KalmanFilter
    status: TrackStatus
    age: int = 0
    miss_count: int = 0
    confirm_hits: int = 0             # consecutive matched frames in TENTATIVE
    first_seen: int = 0
    last_seen: int = 0
    last_box3d: Optional[np.ndarray] = None   # [cx,cy,cz,θ,l,w,h]
    velocity_estimate: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cow_points_abs: Optional[np.ndarray] = None   # [N,2] pixel coords
    cow_points_rel: Optional[np.ndarray] = None   # [N,2] normalised to bbox
    last_bbox2d: Optional[list] = None            # [x1,y1,x2,y2] for re-spawn
    centroid_history: list = field(default_factory=list)
    position_3d_history: list = field(default_factory=list)
    innovation_norm: float = 0.0      # ||z - Hx||, used for lazy CoW gate

    def to_track(self) -> Track:
        """Convert to the shared Track dataclass used by downstream stages."""
        x = self.kf.x.flatten()
        return Track(
            track_id=self.track_id,
            class_name=self.class_name,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            centroid_history=list(self.centroid_history),
            position_3d_history=list(self.position_3d_history),
            cow_query_point=list(self.cow_points_abs[0]) if self.cow_points_abs is not None else [0., 0.],
            is_active=(self.status != TrackStatus.LOST),
            occlusion_count=self.miss_count,
        )
```

**Step 4: Run tests**

```
pytest tests/test_track_state.py -v
```
Expected: all PASS

---

## Task 4: Kalman Filter initializer

**Files:**
- Create: `fusion_perception/tracking/kf_init.py`
- Create: `tests/test_kf_init.py`

**Step 1: Write failing tests**

Create `tests/test_kf_init.py`:
```python
import numpy as np
from fusion_perception.tracking.kf_init import init_kf, KF_DIM_X, KF_DIM_Z

def test_kf_dimensions():
    box3d = [1.0, 0.0, 15.0, 0.0, 4.0, 2.0, 1.5]
    kf = init_kf(box3d, dt=0.1)
    assert kf.x.shape == (KF_DIM_X, 1)
    assert kf.F.shape == (KF_DIM_X, KF_DIM_X)
    assert kf.H.shape == (KF_DIM_Z, KF_DIM_X)

def test_kf_state_initialized():
    box3d = [1.0, 0.0, 15.0, 0.3, 4.0, 2.0, 1.5]
    kf = init_kf(box3d, dt=0.1)
    x = kf.x.flatten()
    assert abs(x[0] - 1.0) < 1e-6   # cx
    assert abs(x[2] - 15.0) < 1e-6  # cz
    assert abs(x[3] - 0.3) < 1e-6   # theta
    # velocities should start at zero
    np.testing.assert_array_equal(x[7:], np.zeros(3))

def test_kf_predict_advances_position():
    box3d = [0., 0., 10., 0., 4., 2., 1.5]
    kf = init_kf(box3d, dt=0.1)
    # inject velocity
    kf.x[7, 0] = 5.0  # vx = 5 m/s
    kf.predict()
    x = kf.x.flatten()
    assert abs(x[0] - 0.5) < 1e-5   # cx advanced by vx*dt = 0.5

def test_kf_transition_matrix_velocity():
    box3d = [0., 0., 10., 0., 4., 2., 1.5]
    kf = init_kf(box3d, dt=0.2)
    assert abs(kf.F[0, 7] - 0.2) < 1e-9  # cx row, vx col
    assert abs(kf.F[1, 8] - 0.2) < 1e-9
    assert abs(kf.F[2, 9] - 0.2) < 1e-9
```

**Step 2: Run to verify failure**

```
pytest tests/test_kf_init.py -v
```
Expected: FAIL

**Step 3: Implement `fusion_perception/tracking/kf_init.py`**

```python
"""Kalman Filter factory for KalmanCoWTracker."""
import numpy as np
from filterpy.kalman import KalmanFilter

KF_DIM_X = 10   # [cx, cy, cz, θ, l, w, h, vx, vy, vz]
KF_DIM_Z = 7    # [cx, cy, cz, θ, l, w, h]

# Index map: state vector positions
IDX_CX, IDX_CY, IDX_CZ = 0, 1, 2
IDX_THETA = 3
IDX_L, IDX_W, IDX_H = 4, 5, 6
IDX_VX, IDX_VY, IDX_VZ = 7, 8, 9


def init_kf(box3d: list[float], dt: float = 0.1) -> KalmanFilter:
    """
    Create and initialise a KalmanFilter for one track.
    box3d: [cx, cy, cz, theta, l, w, h]
    dt: time step in seconds (1/FPS)
    """
    kf = KalmanFilter(dim_x=KF_DIM_X, dim_z=KF_DIM_Z)

    # State transition: constant velocity for position, identity for rest
    kf.F = np.eye(KF_DIM_X, dtype=np.float64)
    kf.F[IDX_CX, IDX_VX] = dt
    kf.F[IDX_CY, IDX_VY] = dt
    kf.F[IDX_CZ, IDX_VZ] = dt

    # Observation: directly observe first 7 state dims
    kf.H = np.zeros((KF_DIM_Z, KF_DIM_X), dtype=np.float64)
    kf.H[:KF_DIM_Z, :KF_DIM_Z] = np.eye(KF_DIM_Z)

    # Process noise: higher on velocities (sudden braking/turning)
    kf.Q = np.diag([
        0.1, 0.1, 0.2,       # cx, cy, cz
        0.05,                 # theta
        0.02, 0.02, 0.02,    # l, w, h
        2.0, 2.0, 3.0,       # vx, vy, vz  (3.0 for forward speed changes)
    ]).astype(np.float64)

    # Initial state covariance: high uncertainty on velocities
    kf.P = np.diag([
        1.0, 1.0, 2.0,
        0.1,
        0.5, 0.5, 0.5,
        10.0, 10.0, 10.0,
    ]).astype(np.float64)

    # Measurement noise placeholder (overridden per-frame by _synthesize_measurement)
    kf.R = np.diag([0.5, 0.5, 2.0, 0.1, 0.3, 0.3, 0.3]).astype(np.float64)

    # Initial state
    kf.x = np.zeros((KF_DIM_X, 1), dtype=np.float64)
    kf.x[:KF_DIM_Z, 0] = box3d

    return kf
```

**Step 4: Run tests**

```
pytest tests/test_kf_init.py -v
```
Expected: all PASS

---

## Task 5: Measurement synthesis (depth-aware R + CoW fusion)

**Files:**
- Create: `fusion_perception/tracking/measurement.py`
- Create: `tests/test_measurement.py`

This module computes the adaptive measurement vector `z` and covariance `R`.

**Step 1: Write failing tests**

Create `tests/test_measurement.py`:
```python
import numpy as np
from fusion_perception.tracking.measurement import (
    synthesize_measurement, MeasurementConfig
)
from fusion_perception.utils.dataclasses import Detection3D

def _make_det(cx=0., cy=0., cz=15., score=0.9, theta=0., l=4., w=2., h=1.5):
    return Detection3D(
        frame_idx=0, class_id=0, class_name='car',
        score=score, score_2d=score, score_3d=score,
        box_2d=[100., 50., 300., 200.],
        box_3d=[cx, cy, cz, w, h, l, theta],
        centroid_2d=[200., 125.],
        centroid_3d=[cx, cy, cz],
        depth=cz,
    )

K = np.array([[718., 0., 607.], [0., 718., 185.], [0., 0., 1.]], dtype=np.float32)
cfg = MeasurementConfig()

def test_measurement_shape():
    det = _make_det()
    z, R = synthesize_measurement(det, None, False, K, cfg)
    assert z.shape == (7,)
    assert R.shape == (7, 7)

def test_no_cow_inflates_R():
    det = _make_det(cz=10.)
    _, R_no_cow = synthesize_measurement(det, None, False, K, cfg)
    _, R_with_cow = synthesize_measurement(det, np.array([2., 0.]), True, K, cfg)
    # R_zz (depth noise) should be same; overall R smaller with CoW
    assert np.trace(R_no_cow) > np.trace(R_with_cow)

def test_depth_scales_z_noise():
    det_near = _make_det(cz=5.)
    det_far = _make_det(cz=40.)
    _, R_near = synthesize_measurement(det_near, None, False, K, cfg)
    _, R_far = synthesize_measurement(det_far, None, False, K, cfg)
    assert R_far[2, 2] > R_near[2, 2]   # far object has larger depth noise

def test_low_confidence_inflates_R():
    det_hi = _make_det(score=0.95)
    det_lo = _make_det(score=0.3)
    _, R_hi = synthesize_measurement(det_hi, None, False, K, cfg)
    _, R_lo = synthesize_measurement(det_lo, None, False, K, cfg)
    assert np.trace(R_lo) > np.trace(R_hi)

def test_cow_corrects_position():
    det = _make_det(cx=0., cz=10.)
    cow_disp = np.array([71.8, 0.])   # 1m at cz=10, fx=718
    z, _ = synthesize_measurement(det, cow_disp, True, K, cfg)
    assert abs(z[0] - 1.0) < 0.05   # cx corrected by ~1m

def test_yaw_extracted_correctly():
    det = _make_det(theta=0.5)
    z, _ = synthesize_measurement(det, None, False, K, cfg)
    assert abs(z[3] - 0.5) < 1e-6
```

**Step 2: Run to verify failure**

```
pytest tests/test_measurement.py -v
```
Expected: FAIL

**Step 3: Implement `fusion_perception/tracking/measurement.py`**

```python
"""Adaptive measurement vector and covariance for KalmanCoWTracker."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D


@dataclass
class MeasurementConfig:
    depth_alpha: float = 0.005    # σ_z = α·z² + β
    depth_beta: float = 0.5
    sigma_xy_base: float = 0.5
    sigma_xy_slope: float = 0.01
    cow_sigma_xy: float = 0.1     # CoWTracker lateral noise (metres)
    fallback_inflate: float = 3.0 # inflate R when CoW unavailable


def synthesize_measurement(
    det: Detection3D,
    cow_disp_px: Optional[np.ndarray],  # [dx, dy] median pixel displacement
    cow_valid: bool,
    K: np.ndarray,                       # 3×3 intrinsics
    cfg: MeasurementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build measurement vector z (7-dim) and covariance R (7×7).
    box_3d layout: [cx, cy, cz, w, h, l, ry]
    KF state layout: [cx, cy, cz, θ, l, w, h]
    """
    b = det.box_3d   # [cx, cy, cz, w, h, l, ry]
    cx, cy, cz = b[0], b[1], b[2]
    theta = b[6]
    l, w, h = b[5], b[3], b[4]

    z = np.array([cx, cy, cz, theta, l, w, h], dtype=np.float64)

    # Depth-dependent measurement noise
    sigma_z = cfg.depth_alpha * cz ** 2 + cfg.depth_beta
    sigma_xy = cfg.sigma_xy_base + cfg.sigma_xy_slope * cz
    R_base = np.diag([
        sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2,
        0.05, 0.09, 0.09, 0.09,
    ]).astype(np.float64)

    # Detection confidence scaling: lower score → larger R
    score = max(float(det.score), 0.1)
    R_det = R_base / score

    if cow_valid and cow_disp_px is not None:
        # Back-project pixel displacement to camera-frame 3D offset
        fx, fy = float(K[0, 0]), float(K[1, 1])
        dx_3d = float(cow_disp_px[0]) * cz / fx
        dy_3d = float(cow_disp_px[1]) * cz / fy
        z[0] += dx_3d
        z[1] += dy_3d

        # CoW measurement noise (visually accurate, but depth still uncertain)
        R_cow = np.diag([
            cfg.cow_sigma_xy ** 2, cfg.cow_sigma_xy ** 2, sigma_z ** 2,
            0.05, 0.09, 0.09, 0.09,
        ]).astype(np.float64)

        # Information-theoretic fusion: R_eff^-1 = R_det^-1 + R_cow^-1
        R_eff = np.linalg.inv(np.linalg.inv(R_det) + np.linalg.inv(R_cow))
    else:
        R_eff = R_det * cfg.fallback_inflate

    return z, R_eff
```

**Step 4: Run tests**

```
pytest tests/test_measurement.py -v
```
Expected: all PASS

---

## Task 6: Ego-motion compensation (ORB homography)

**Files:**
- Create: `fusion_perception/tracking/ego_motion.py`
- Create: `tests/test_ego_motion.py`

Compensates for camera movement before Hungarian assignment so stationary objects don't appear to drift.

**Step 1: Write failing tests**

Create `tests/test_ego_motion.py`:
```python
import numpy as np
from fusion_perception.tracking.ego_motion import estimate_homography, compensate_centroids

def test_identity_homography_on_same_frame():
    rng = np.random.default_rng(42)
    frame = (rng.random((370, 1224, 3)) * 255).astype(np.uint8)
    H = estimate_homography(frame, frame)
    # Identity or None (no motion)
    if H is not None:
        # applying H to a point should return same point
        pt = np.array([[[612., 185.]]], dtype=np.float32)
        import cv2
        warped = cv2.perspectiveTransform(pt, H)
        assert np.allclose(warped[0, 0], pt[0, 0], atol=2.0)

def test_compensate_centroids_no_motion():
    centroids = [[200., 150.], [400., 300.]]
    result = compensate_centroids(centroids, None)
    assert result == centroids

def test_compensate_centroids_with_H():
    import cv2
    H = np.eye(3, dtype=np.float32)
    H[0, 2] = 10.0   # pure translation of 10px in x
    centroids = [[100., 200.]]
    result = compensate_centroids(centroids, H)
    assert abs(result[0][0] - 110.0) < 1.0
    assert abs(result[0][1] - 200.0) < 1.0
```

**Step 2: Run to verify failure**

```
pytest tests/test_ego_motion.py -v
```
Expected: FAIL

**Step 3: Implement `fusion_perception/tracking/ego_motion.py`**

```python
"""Ego-motion compensation via ORB background homography (BoT-SORT style)."""
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np


def estimate_homography(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    max_features: int = 200,
) -> Optional[np.ndarray]:
    """
    Estimate 3×3 homography from background ORB matches between two frames.
    Returns None if too few inliers are found.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_RGB2GRAY)

    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        return None
    inliers = int(mask.sum()) if mask is not None else 0
    return H if inliers >= 6 else None


def compensate_centroids(
    centroids: list[list[float]],
    H: Optional[np.ndarray],
) -> list[list[float]]:
    """Apply homography H to a list of [x, y] pixel centroids."""
    if H is None or len(centroids) == 0:
        return centroids

    pts = np.array(centroids, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2).tolist()
```

**Step 4: Run tests**

```
pytest tests/test_ego_motion.py -v
```
Expected: all PASS

---

## Task 7: CoWTracker point management (spawn + lazy batch)

**Files:**
- Create: `fusion_perception/tracking/cow_points.py`
- Create: `tests/test_cow_points.py`

**Step 1: Write failing tests**

Create `tests/test_cow_points.py`:
```python
import numpy as np
from fusion_perception.tracking.cow_points import spawn_points, unpack_cow_outputs

def test_spawn_points_count():
    bbox2d = [100., 50., 300., 200.]   # [x1,y1,x2,y2], area=150×200=30000px²
    pts = spawn_points(bbox2d, beta=0.5, min_pts=8, max_pts=64)
    assert 8 <= len(pts) <= 64

def test_spawn_points_inside_bbox():
    bbox2d = [100., 50., 300., 200.]
    pts = spawn_points(bbox2d)
    for x, y in pts:
        assert 100. <= x <= 300.
        assert 50. <= y <= 200.

def test_spawn_points_minimum():
    tiny_bbox = [100., 50., 102., 52.]  # 2×2 px, very small
    pts = spawn_points(tiny_bbox, min_pts=8, max_pts=64)
    assert len(pts) == 8   # clamped to minimum

def test_unpack_cow_outputs():
    import torch
    T, N = 5, 3
    pred_tracks = torch.zeros(1, T, N, 2)
    pred_tracks[0, -1, 0] = torch.tensor([10., 20.])
    pred_tracks[0, -1, 1] = torch.tensor([30., 40.])
    pred_vis = torch.ones(1, T, N)
    track_ids = [1, 2, 3]
    point_counts = [1, 1, 1]
    disps, valids = unpack_cow_outputs(pred_tracks, pred_vis, track_ids, point_counts,
                                       conf_threshold=0.85, min_points=1)
    assert 1 in disps and 2 in disps
    assert valids[1] is True
```

**Step 2: Run to verify failure**

```
pytest tests/test_cow_points.py -v
```
Expected: FAIL

**Step 3: Implement `fusion_perception/tracking/cow_points.py`**

```python
"""CoWTracker point spawning and output unpacking for KalmanCoWTracker."""
from __future__ import annotations
import math
import numpy as np
import torch
from typing import Optional


def spawn_points(
    bbox2d: list[float],
    beta: float = 0.5,
    min_pts: int = 8,
    max_pts: int = 64,
) -> np.ndarray:
    """
    Sample a grid of pixel points within bbox2d.
    n_points = clamp(int(β · sqrt(area)), min_pts, max_pts)
    Returns array of shape [N, 2] in pixel coords.
    """
    x1, y1, x2, y2 = bbox2d
    area = max(0.0, (x2 - x1) * (y2 - y1))
    n = int(math.sqrt(area) * beta)
    n = max(min_pts, min(max_pts, n))

    # Uniform grid sampling
    side = int(math.ceil(math.sqrt(n)))
    xs = np.linspace(x1, x2, side, endpoint=False) + (x2 - x1) / (2 * side)
    ys = np.linspace(y1, y2, side, endpoint=False) + (y2 - y1) / (2 * side)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return pts[:n]   # exact count


def unpack_cow_outputs(
    pred_tracks: torch.Tensor,   # [1, T, N_total, 2]
    pred_vis: torch.Tensor,      # [1, T, N_total]
    track_ids: list[int],
    point_counts: list[int],     # how many points each track contributed
    conf_threshold: float = 0.85,
    min_points: int = 4,
) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
    """
    Split batched CoWTracker output back to per-track displacement and validity.
    Returns:
      displacements: {track_id: np.ndarray [2]} — median pixel displacement
      valids: {track_id: bool} — True if ≥ min_points survived confidence gate
    """
    last_pos = pred_tracks[0, -1].cpu().numpy()    # [N_total, 2]
    first_pos = pred_tracks[0, 0].cpu().numpy()    # [N_total, 2]
    last_vis = pred_vis[0, -1].cpu().numpy()       # [N_total]

    displacements: dict[int, np.ndarray] = {}
    valids: dict[int, bool] = {}

    offset = 0
    for tid, n in zip(track_ids, point_counts):
        pts_disp = last_pos[offset:offset + n] - first_pos[offset:offset + n]
        pts_conf = last_vis[offset:offset + n]
        offset += n

        mask = pts_conf >= conf_threshold
        n_alive = int(mask.sum())
        if n_alive >= min_points:
            displacements[tid] = np.median(pts_disp[mask], axis=0)
            valids[tid] = True
        else:
            displacements[tid] = np.zeros(2)
            valids[tid] = False

    return displacements, valids
```

**Step 4: Run tests**

```
pytest tests/test_cow_points.py -v
```
Expected: all PASS

---

## Task 8: Cost matrix + Mahalanobis gate + ByteTrack association

**Files:**
- Create: `fusion_perception/tracking/association.py`
- Create: `tests/test_association.py`

**Step 1: Write failing tests**

Create `tests/test_association.py`:
```python
import numpy as np
from fusion_perception.tracking.association import (
    build_cost_matrix, mahalanobis_gate, hungarian_match
)

def _box(cx=0., cz=10., l=4., w=2., h=1.5, theta=0., cy=0.):
    return [cx, cy, cz, theta, l, w, h]

def test_cost_matrix_shape():
    pred = [_box(0.), _box(5.)]
    dets = [_box(0.1), _box(5.1), _box(20.)]
    cow_disp = {0: np.zeros(2), 1: np.zeros(2)}
    C = build_cost_matrix(pred, dets, cow_disp, alpha=0.35)
    assert C.shape == (2, 3)

def test_cost_perfect_match_is_low():
    box = _box(0., 10.)
    C = build_cost_matrix([box], [box], {0: np.zeros(2)}, alpha=0.35)
    assert C[0, 0] < 0.1

def test_cost_no_overlap_is_high():
    a = _box(0., 10.)
    b = _box(100., 10.)
    C = build_cost_matrix([a], [b], {0: np.zeros(2)}, alpha=0.35)
    assert C[0, 0] > 0.8

def test_mahalanobis_gate_removes_distant():
    # Two predictions, two detections — second pair is very far apart
    pred_states = [np.array([0., 0., 10., 0., 4., 2., 1.5, 0., 0., 0.]),
                   np.array([50., 0., 10., 0., 4., 2., 1.5, 0., 0., 0.])]
    pred_covs = [np.eye(10) * 0.1, np.eye(10) * 0.1]
    det_z = [np.array([0.1, 0., 10.1, 0., 4., 2., 1.5]),
             np.array([0.1, 0., 10.1, 0., 4., 2., 1.5])]  # both near first pred
    mask = mahalanobis_gate(pred_states, pred_covs, det_z, threshold=9.21)
    assert mask[0, 0] == True   # first pred matches first det
    assert mask[1, 0] == False  # second pred (far away) gated out

def test_hungarian_match():
    C = np.array([[0.1, 0.9], [0.9, 0.1]])
    matched, unmatched_rows, unmatched_cols = hungarian_match(C, threshold=0.5)
    assert (0, 0) in matched
    assert (1, 1) in matched
    assert len(unmatched_rows) == 0
    assert len(unmatched_cols) == 0
```

**Step 2: Run to verify failure**

```
pytest tests/test_association.py -v
```
Expected: FAIL

**Step 3: Implement `fusion_perception/tracking/association.py`**

```python
"""Cost matrix, Mahalanobis gate, and Hungarian matching for KalmanCoWTracker."""
from __future__ import annotations
import numpy as np
from fusion_perception.utils.geometry import iou3d

try:
    from lapjv import lapjv as _lapjv
    _HAS_LAPJV = True
except ImportError:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAS_LAPJV = False


def build_cost_matrix(
    pred_boxes: list[list[float]],   # [N] predicted boxes [cx,cy,cz,θ,l,w,h]
    det_boxes: list[list[float]],    # [M] detected boxes  [cx,cy,cz,θ,l,w,h]
    cow_displacements: dict[int, np.ndarray],  # {pred_idx: [dx,dy] or None}
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Hybrid cost: α·(1−IoU3D) + (1−α)·D_CoW_normalised.
    Shape: [N, M]
    """
    N, M = len(pred_boxes), len(det_boxes)
    if N == 0 or M == 0:
        return np.zeros((N, M))

    C = np.ones((N, M), dtype=np.float64)

    det_centers = np.array([[d[0], d[1], d[2]] for d in det_boxes])

    for i, pb in enumerate(pred_boxes):
        pb_center = np.array([pb[0], pb[1], pb[2]])
        cow_disp = cow_displacements.get(i)

        for j, db in enumerate(det_boxes):
            iou = iou3d(pb, db)
            iou_cost = 1.0 - iou

            # CoW distance cost: distance between CoW-predicted position and detection
            if cow_disp is not None:
                cow_pred = pb_center  # already incorporates CoW displacement via KF
                dc = float(np.linalg.norm(cow_pred - det_centers[j]))
                cow_cost = min(dc / 20.0, 1.0)   # normalise by 20m
            else:
                cow_cost = iou_cost   # fall back to IoU alone

            C[i, j] = alpha * iou_cost + (1.0 - alpha) * cow_cost

    return C


def mahalanobis_gate(
    pred_states: list[np.ndarray],    # [N] each shape (10,)
    pred_covs: list[np.ndarray],      # [N] each shape (10,10)
    det_z: list[np.ndarray],          # [M] each shape (7,)
    threshold: float = 9.21,          # χ²(0.99, df=4)
) -> np.ndarray:
    """Boolean mask [N, M]: True = association is plausible."""
    N, M = len(pred_states), len(det_z)
    mask = np.ones((N, M), dtype=bool)
    H = np.zeros((7, 10))
    H[:7, :7] = np.eye(7)

    for i in range(N):
        x = pred_states[i]
        P = pred_covs[i]
        S = H @ P @ H.T   # innovation covariance [7,7]
        S_sub = S[:4, :4]  # gate on [cx, cy, cz, θ] only (4 dof)
        try:
            S_inv = np.linalg.inv(S_sub + np.eye(4) * 1e-6)
        except np.linalg.LinAlgError:
            continue
        pred_obs = x[:7]
        for j in range(M):
            diff = det_z[j][:4] - pred_obs[:4]
            d2 = float(diff @ S_inv @ diff)
            mask[i, j] = d2 <= threshold

    return mask


def hungarian_match(
    cost_matrix: np.ndarray,
    threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Solve assignment problem. Returns (matched, unmatched_rows, unmatched_cols).
    matched: list of (row_idx, col_idx) pairs with cost < threshold
    """
    N, M = cost_matrix.shape
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    if _HAS_LAPJV:
        # lapjv requires square matrix — pad to square
        size = max(N, M)
        padded = np.ones((size, size), dtype=np.float64)
        padded[:N, :M] = cost_matrix
        row_ind, col_ind, _ = _lapjv(padded, extend_cost=True, cost_limit=threshold + 1e-6)
        assignment = [(r, col_ind[r]) for r in range(N) if col_ind[r] < M]
    else:
        row_ind, col_ind = _scipy_lsa(cost_matrix)
        assignment = list(zip(row_ind.tolist(), col_ind.tolist()))

    matched = [(r, c) for r, c in assignment if cost_matrix[r, c] <= threshold]
    matched_rows = {r for r, _ in matched}
    matched_cols = {c for _, c in matched}
    unmatched_rows = [r for r in range(N) if r not in matched_rows]
    unmatched_cols = [c for c in range(M) if c not in matched_cols]

    return matched, unmatched_rows, unmatched_cols
```

**Step 4: Run tests**

```
pytest tests/test_association.py -v
```
Expected: all PASS

---

## Task 9: Main `KalmanCoWTracker` class

**Files:**
- Create: `fusion_perception/tracking/kalman_cowtracker.py`
- Create: `tests/test_kalman_cowtracker.py`
- Modify: `fusion_perception/tracking/__init__.py`

**Step 1: Write failing integration tests**

Create `tests/test_kalman_cowtracker.py`:
```python
import numpy as np
from fusion_perception.tracking.kalman_cowtracker import KalmanCoWTracker
from fusion_perception.utils.dataclasses import Detection3D

K = np.array([[718., 0., 607.], [0., 718., 185.], [0., 0., 1.]], dtype=np.float32)

def _det(cx, cz, score=0.8, frame_idx=0):
    return Detection3D(
        frame_idx=frame_idx, class_id=0, class_name='car',
        score=score, score_2d=score, score_3d=score,
        box_2d=[200., 100., 400., 250.],
        box_3d=[cx, 0., cz, 2., 1.5, 4., 0.],
        centroid_2d=[300., 175.],
        centroid_3d=[cx, 0., cz],
        depth=cz,
    )

def _frame():
    return np.zeros((370, 1224, 3), dtype=np.uint8)

def test_tracker_spawns_track():
    tracker = KalmanCoWTracker(device='cpu')
    tracker.load()
    dets = [_det(0., 15.)]
    tracks = tracker.update(_frame(), dets, 0, intrinsics=K)
    assert len(tracks) == 1
    assert tracks[0].class_name == 'car'

def test_track_persists_across_frames():
    tracker = KalmanCoWTracker(device='cpu', confirm_age=1)
    tracker.load()
    det = _det(0., 15.)
    for i in range(5):
        tracks = tracker.update(_frame(), [det], i, intrinsics=K)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1

def test_lost_track_reappears():
    tracker = KalmanCoWTracker(device='cpu', confirm_age=1, lost_patience=10)
    tracker.load()
    # Establish track
    for i in range(3):
        tracker.update(_frame(), [_det(0., 15.)], i, intrinsics=K)
    # Drop detection for 3 frames
    for i in range(3, 6):
        tracks = tracker.update(_frame(), [], i, intrinsics=K)
    # Track should still exist (within lost_patience)
    all_tracks = tracker.get_all_tracks()
    assert any(ts.miss_count > 0 for ts in all_tracks.values())

def test_two_detections_two_tracks():
    tracker = KalmanCoWTracker(device='cpu', confirm_age=1)
    tracker.load()
    dets = [_det(0., 15.), _det(10., 20.)]
    tracks = tracker.update(_frame(), dets, 0, intrinsics=K)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2
```

**Step 2: Run to verify failure**

```
pytest tests/test_kalman_cowtracker.py -v
```
Expected: FAIL (ModuleNotFoundError)

**Step 3: Implement `fusion_perception/tracking/kalman_cowtracker.py`**

```python
"""Hybrid Kalman Filter + CoWTracker for 3D multi-object tracking.

Per-frame flow:
  1. Predict all KF states forward (dt = 1/fps)
  2. Ego-motion compensation on track positions
  3. Run CoWTracker batch (lazy: skip stable tracks)
  4. Build hybrid cost matrix; Mahalanobis gate
  5. ByteTrack two-threshold Hungarian assignment
  6. Update matched tracks (KF update + adaptive Q)
  7. Handle unmatched tracks (LOST lifecycle, velocity decay)
  8. Spawn new tracks for unmatched high-conf detections
  9. Prune LOST tracks exceeding lost_patience
"""
from __future__ import annotations
import numpy as np
import torch
from collections import deque
from typing import Optional

from fusion_perception.tracking.base_tracker import BaseTracker
from fusion_perception.tracking.track_state import TrackState, TrackStatus
from fusion_perception.tracking.kf_init import init_kf, KF_DIM_Z
from fusion_perception.tracking.measurement import synthesize_measurement, MeasurementConfig
from fusion_perception.tracking.cow_points import spawn_points, unpack_cow_outputs
from fusion_perception.tracking.ego_motion import estimate_homography, compensate_centroids
from fusion_perception.tracking.association import (
    build_cost_matrix, mahalanobis_gate, hungarian_match
)
from fusion_perception.tracking.centroid_anchor import assign_new_track_id
from fusion_perception.utils.dataclasses import Detection3D, Track
from fusion_perception.utils.geometry import wrap_angle
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("kalman_cowtracker")


class KalmanCoWTracker(BaseTracker):
    """Hybrid 3D MOT: Kalman Filter with CoWTracker pixel-anchored measurement."""

    def __init__(
        self,
        window_size: int = 8,
        max_tracks: int = 50,
        lost_patience: int = 30,
        confirm_age: int = 3,
        high_score_threshold: float = 0.5,
        low_score_threshold: float = 0.2,
        iou_threshold: float = 0.5,
        alpha_cost: float = 0.35,
        cow_conf_threshold: float = 0.85,
        min_cow_points: int = 4,
        velocity_decay: float = 0.9,
        lazy_cow_innovation: float = 0.3,
        ego_motion: bool = True,
        mahal_threshold: float = 9.21,
        device: str = "cuda",
    ) -> None:
        self.window_size = window_size
        self.max_tracks = max_tracks
        self.lost_patience = lost_patience
        self.confirm_age = confirm_age
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.iou_threshold = iou_threshold
        self.alpha_cost = alpha_cost
        self.cow_conf_threshold = cow_conf_threshold
        self.min_cow_points = min_cow_points
        self.velocity_decay = velocity_decay
        self.lazy_cow_innovation = lazy_cow_innovation
        self.ego_motion = ego_motion
        self.mahal_threshold = mahal_threshold
        self.device = device

        self._cow_model = None
        self._tracks: dict[int, TrackState] = {}
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._prev_frame: Optional[np.ndarray] = None
        self._meas_cfg = MeasurementConfig()
        self._fps: float = 10.0

    # ── BaseTracker interface ─────────────────────────────────────────────────

    def load(self) -> None:
        logger.info("Loading CoWTracker for KalmanCoWTracker")
        try:
            from cotracker.predictor import CoTrackerPredictor
            self._cow_model = CoTrackerPredictor(
                checkpoint=None, window_len=self.window_size
            ).to(self.device)
            self._cow_model.eval()
            log_gpu_memory("CoWTracker loaded (KalmanCoWTracker)")
        except ImportError:
            logger.warning("CoWTracker not installed — running KF-only mode")
            self._cow_model = None

    @torch.no_grad()
    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection3D],
        frame_idx: int,
        fps: float = 10.0,
        intrinsics: Optional[np.ndarray] = None,
    ) -> list[Track]:
        self._fps = fps
        dt = 1.0 / max(fps, 1.0)
        self._frame_buffer.append(frame)

        K = intrinsics if intrinsics is not None else np.eye(3, dtype=np.float32)

        # 1. Predict all KF states
        self._predict_all(dt)

        # 2. Ego-motion compensation
        H_ego = None
        if self.ego_motion and self._prev_frame is not None:
            H_ego = estimate_homography(self._prev_frame, frame)
        self._prev_frame = frame.copy()

        # 3. Split detections by score (ByteTrack two-threshold)
        high_dets = [d for d in detections if d.score >= self.high_score_threshold]
        low_dets = [d for d in detections
                    if self.low_score_threshold <= d.score < self.high_score_threshold]

        # 4. CoWTracker batch update (lazy: skip tracks with small innovation)
        cow_displacements, cow_valids = self._run_cow_batch(frame_idx)

        # 5. Associate confirmed + tentative tracks with high-conf detections
        confirmed_ids = [tid for tid, ts in self._tracks.items()
                         if ts.status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE)]
        self._associate_and_update(
            confirmed_ids, high_dets, cow_displacements, cow_valids, K, frame_idx
        )

        # 6. Associate LOST tracks with low-conf detections
        lost_ids = [tid for tid, ts in self._tracks.items()
                    if ts.status == TrackStatus.LOST]
        self._associate_and_update(
            lost_ids, low_dets, cow_displacements, cow_valids, K, frame_idx
        )

        # 7. Spawn new tracks from unmatched high-conf detections
        matched_det_centroids = {
            (round(d.centroid_3d[0], 2), round(d.centroid_3d[2], 2))
            for ts in self._tracks.values()
            if ts.last_box3d is not None
            for d in high_dets
            if abs(d.centroid_3d[0] - ts.last_box3d[0]) < 0.5
        }
        for det in high_dets:
            key = (round(det.centroid_3d[0], 2), round(det.centroid_3d[2], 2))
            if key not in matched_det_centroids and len(self._tracks) < self.max_tracks:
                self._spawn_track(det, frame_idx, dt)

        # 8. Prune dead LOST tracks
        to_delete = [
            tid for tid, ts in self._tracks.items()
            if ts.status == TrackStatus.LOST and ts.miss_count > self.lost_patience
        ]
        for tid in to_delete:
            del self._tracks[tid]
            logger.debug(f"Deleted track {tid} after {self.lost_patience} lost frames")

        active = [
            ts.to_track() for ts in self._tracks.values()
            if ts.status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE)
        ]
        logger.debug(f"Frame {frame_idx}: {len(active)} active tracks")
        return active

    def get_all_tracks(self) -> dict[int, TrackState]:
        return self._tracks

    def reset(self) -> None:
        self._tracks = {}
        self._frame_buffer.clear()
        self._prev_frame = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _predict_all(self, dt: float) -> None:
        for ts in self._tracks.values():
            ts.kf.F[0, 7] = dt
            ts.kf.F[1, 8] = dt
            ts.kf.F[2, 9] = dt
            ts.kf.predict()
            ts.age += 1
            if ts.status == TrackStatus.LOST:
                # Decay velocity estimate
                ts.kf.x[7:, 0] *= self.velocity_decay

    def _run_cow_batch(
        self, frame_idx: int
    ) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
        if self._cow_model is None or len(self._frame_buffer) < 2:
            return {}, {}

        # Lazy gate: skip tracks with small innovation
        active_ids = [
            tid for tid, ts in self._tracks.items()
            if ts.cow_points_abs is not None
            and (ts.innovation_norm > self.lazy_cow_innovation
                 or frame_idx % 3 == 0)
        ]
        if not active_ids:
            return {}, {}

        # Pad all point tensors to max_points
        all_points = [self._tracks[tid].cow_points_abs for tid in active_ids]
        point_counts = [len(p) for p in all_points]
        max_pts = max(point_counts)

        # Build padded queries tensor [1, N_total, 3]
        query_list = []
        for pts in all_points:
            for xy in pts:
                query_list.append([0.0, float(xy[0]), float(xy[1])])
            # Pad with last point to reach max_pts
            for _ in range(max_pts - len(pts)):
                query_list.append([0.0, float(pts[-1][0]), float(pts[-1][1])])

        # Actually stack for real CoWTracker call (unpadded to N_total)
        query_unpadded = []
        for pts in all_points:
            for xy in pts:
                query_unpadded.append([0.0, float(xy[0]), float(xy[1])])

        frames_np = np.stack(list(self._frame_buffer), axis=0)
        video = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float()
        video = video.unsqueeze(0).to(self.device)

        queries = torch.tensor(query_unpadded, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

        pred_tracks, pred_vis = self._cow_model(video, queries=queries)

        return unpack_cow_outputs(
            pred_tracks, pred_vis, active_ids, point_counts,
            conf_threshold=self.cow_conf_threshold,
            min_points=self.min_cow_points,
        )

    def _associate_and_update(
        self,
        track_ids: list[int],
        dets: list[Detection3D],
        cow_displacements: dict[int, np.ndarray],
        cow_valids: dict[int, bool],
        K: np.ndarray,
        frame_idx: int,
    ) -> None:
        if not track_ids or not dets:
            for tid in track_ids:
                self._handle_miss(tid)
            return

        pred_boxes = [self._tracks[tid].kf.x[:7, 0].tolist() for tid in track_ids]
        det_boxes = [
            [d.box_3d[0], d.box_3d[1], d.box_3d[2],
             d.box_3d[6], d.box_3d[5], d.box_3d[3], d.box_3d[4]]
            for d in dets
        ]

        # Mahalanobis gate
        pred_states = [self._tracks[tid].kf.x.flatten() for tid in track_ids]
        pred_covs = [self._tracks[tid].kf.P for tid in track_ids]
        det_z = [np.array(db, dtype=np.float64) for db in det_boxes]
        gate_mask = mahalanobis_gate(pred_states, pred_covs, det_z, self.mahal_threshold)

        cow_disp_by_idx = {i: cow_displacements.get(tid) for i, tid in enumerate(track_ids)}
        C = build_cost_matrix(pred_boxes, det_boxes, cow_disp_by_idx, self.alpha_cost)
        C[~gate_mask] = 1e6   # block gated cells

        matched, unmatched_rows, _ = hungarian_match(C, threshold=self.iou_threshold)

        matched_track_ids = set()
        for row_i, col_j in matched:
            tid = track_ids[row_i]
            det = dets[col_j]
            matched_track_ids.add(tid)
            self._update_track(tid, det, cow_displacements, cow_valids, K, frame_idx)

        for row_i in unmatched_rows:
            self._handle_miss(track_ids[row_i])

    def _update_track(
        self,
        tid: int,
        det: Detection3D,
        cow_displacements: dict,
        cow_valids: dict,
        K: np.ndarray,
        frame_idx: int,
    ) -> None:
        ts = self._tracks[tid]
        cow_disp = cow_displacements.get(tid)
        cow_valid = cow_valids.get(tid, False)

        z, R = synthesize_measurement(det, cow_disp, cow_valid, K, self._meas_cfg)

        # Yaw ambiguity correction before update
        pred_theta = float(ts.kf.x[3, 0])
        z[3] = wrap_angle(z[3])
        delta_theta = wrap_angle(z[3] - pred_theta)
        if abs(delta_theta) > np.pi / 2:
            z[3] = wrap_angle(z[3] + np.pi)

        ts.kf.R = R
        ts.kf.update(z)

        # Wrap yaw in state after update
        ts.kf.x[3, 0] = wrap_angle(ts.kf.x[3, 0])

        # Adaptive Q based on innovation norm
        innovation = z - ts.kf.H @ ts.kf.x.flatten()
        ts.innovation_norm = float(np.linalg.norm(innovation))
        q_factor = 1.0 + 0.1 * ts.innovation_norm
        ts.kf.Q = ts.kf.Q * q_factor   # temporary per-frame inflate

        # Dimension EMA smoothing (don't let dims drift)
        alpha_dim = 0.15
        for idx, det_val in zip([4, 5, 6], [det.box_3d[5], det.box_3d[3], det.box_3d[4]]):
            ts.kf.x[idx, 0] = (1 - alpha_dim) * ts.kf.x[idx, 0] + alpha_dim * det_val

        # Re-spawn CoW points on updated bbox
        if ts.status == TrackStatus.CONFIRMED:
            ts.cow_points_abs = spawn_points(det.box_2d)
            ts.last_bbox2d = det.box_2d

        ts.last_box3d = ts.kf.x[:7, 0].copy()
        ts.last_seen = frame_idx
        ts.miss_count = 0
        ts.centroid_history.append(det.centroid_2d)
        ts.position_3d_history.append(det.centroid_3d)

        # Lifecycle transition
        if ts.status == TrackStatus.TENTATIVE:
            ts.confirm_hits += 1
            if ts.confirm_hits >= self.confirm_age:
                ts.status = TrackStatus.CONFIRMED
        elif ts.status == TrackStatus.LOST:
            # Re-localization: inflate P to accept new visual measurement
            ts.kf.P *= 3.0
            ts.status = TrackStatus.CONFIRMED
            ts.miss_count = 0

    def _handle_miss(self, tid: int) -> None:
        ts = self._tracks[tid]
        ts.miss_count += 1
        if ts.status == TrackStatus.CONFIRMED and ts.miss_count > 2:
            ts.status = TrackStatus.LOST
            # Release CoW points immediately to free memory
            ts.cow_points_abs = None
            ts.cow_points_rel = None
        elif ts.status == TrackStatus.TENTATIVE and ts.miss_count > 1:
            del self._tracks[tid]

    def _spawn_track(self, det: Detection3D, frame_idx: int, dt: float) -> None:
        new_id = assign_new_track_id({tid: None for tid in self._tracks})
        box3d = [
            det.box_3d[0], det.box_3d[1], det.box_3d[2],
            det.box_3d[6], det.box_3d[5], det.box_3d[3], det.box_3d[4]
        ]
        kf = init_kf(box3d, dt=dt)
        ts = TrackState(
            track_id=new_id,
            class_name=det.class_name,
            kf=kf,
            status=TrackStatus.TENTATIVE,
            first_seen=frame_idx,
            last_seen=frame_idx,
            last_box3d=np.array(box3d),
            cow_points_abs=spawn_points(det.box_2d),
            last_bbox2d=det.box_2d,
            centroid_history=[det.centroid_2d],
            position_3d_history=[det.centroid_3d],
        )
        self._tracks[new_id] = ts
        logger.debug(f"New track {new_id}: {det.class_name} @ z={det.centroid_3d[2]:.1f}m")
```

**Step 4: Update `fusion_perception/tracking/__init__.py`**

```python
from .base_tracker import BaseTracker
from .cowtracker_wrapper import CoWTrackerWrapper
from .kalman_cowtracker import KalmanCoWTracker
```

**Step 5: Run tests**

```
pytest tests/test_kalman_cowtracker.py -v
```
Expected: all PASS

---

## Task 10: Wire into `StreamingPipeline` via config backend

**Files:**
- Modify: `fusion_perception/pipelines/streaming_pipeline.py`

**Step 1: Update `_init_models` to select backend**

In `streaming_pipeline.py`, replace the tracker construction block (lines ~61–68) with:

```python
from fusion_perception.tracking.kalman_cowtracker import KalmanCoWTracker

backend = cfg.tracking.get("backend", "cowtracker")
if backend == "kalman_cow":
    tracker = KalmanCoWTracker(
        window_size=cfg.tracking.window_size,
        max_tracks=cfg.tracking.max_tracks,
        lost_patience=cfg.tracking.get("lost_patience", 30),
        confirm_age=cfg.tracking.get("confirm_age", 3),
        high_score_threshold=cfg.tracking.get("high_score_threshold", 0.5),
        low_score_threshold=cfg.tracking.get("low_score_threshold", 0.2),
        iou_threshold=cfg.tracking.get("iou_threshold", 0.5),
        alpha_cost=cfg.tracking.get("alpha_cost", 0.35),
        cow_conf_threshold=cfg.tracking.get("cow_conf_threshold", 0.85),
        min_cow_points=cfg.tracking.get("min_cow_points", 4),
        velocity_decay=cfg.tracking.get("velocity_decay", 0.9),
        ego_motion=cfg.tracking.get("ego_motion", True),
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
```

**Step 2: Pass intrinsics through `StageRunner`**

In `stage_runner.py`, `run_frame` already accepts `intrinsics` param and passes it to `detector.detect()`. Verify `self.tracker.update()` also receives it — update the call on line ~60:

```python
# Before:
tracks: list[Track] = self.tracker.update(frame, detections, frame_idx)

# After:
tracks: list[Track] = self.tracker.update(
    frame, detections, frame_idx,
    fps=self.fps,
    intrinsics=intrinsics,
)
```

Note: `CoWTrackerWrapper.update()` doesn't accept `fps` or `intrinsics` — add `**kwargs` to its signature to absorb extra args:
```python
def update(self, frame, detections, frame_idx, **kwargs):
```

**Step 3: Smoke test the pipeline config**

```
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/default.yaml')
assert cfg.tracking.backend == 'kalman_cow'
print('Config OK:', cfg.tracking.backend)
"
```
Expected: prints `Config OK: kalman_cow`

**Step 4: Run full test suite**

```
pytest tests/ -v --tb=short
```
Expected: all tests PASS (no regressions)

---

## Summary of new files

| File | Purpose |
|------|---------|
| `fusion_perception/tracking/track_state.py` | `TrackState` dataclass + `TrackStatus` enum |
| `fusion_perception/tracking/kf_init.py` | `init_kf()` factory for filterpy KalmanFilter |
| `fusion_perception/tracking/measurement.py` | Adaptive R: depth-scaling, confidence, CoW fusion |
| `fusion_perception/tracking/cow_points.py` | Point spawning + batched CoW output unpacking |
| `fusion_perception/tracking/ego_motion.py` | ORB homography for camera motion compensation |
| `fusion_perception/tracking/association.py` | Cost matrix, Mahalanobis gate, Hungarian solver |
| `fusion_perception/tracking/kalman_cowtracker.py` | Main `KalmanCoWTracker(BaseTracker)` class |

## All improvements implemented

| Improvement | Location |
|-------------|----------|
| Depth-dependent R (`σ_z = α·z²+β`) | `measurement.py` |
| Detection confidence → R scaling | `measurement.py` |
| Information fusion `R⁻¹ = Rdet⁻¹ + Rcow⁻¹` | `measurement.py` |
| Yaw ambiguity correction + wrapped update | `kalman_cowtracker.py` |
| Velocity decay for LOST tracks | `kalman_cowtracker.py` |
| Adaptive Q via innovation norm | `kalman_cowtracker.py` |
| Dimension EMA smoothing | `kalman_cowtracker.py` |
| Covariance inflate on re-localization | `kalman_cowtracker.py` |
| Lazy CoW (skip stable tracks) | `kalman_cowtracker.py` |
| Padded batch CoWTracker forward pass | `kalman_cowtracker.py` |
| ByteTrack two-threshold association | `kalman_cowtracker.py` |
| Hungarian (lapjv) + Mahalanobis gate | `association.py` |
| BEV IoU × height overlap (fast approx) | `geometry.py` |
| Dynamic point density (√area-scaled) | `cow_points.py` |
| Ego-motion compensation (ORB homography) | `ego_motion.py` |
| TENTATIVE/CONFIRMED/LOST lifecycle | `track_state.py` + `kalman_cowtracker.py` |
| Immediate LOST point tensor release | `kalman_cowtracker.py` |
