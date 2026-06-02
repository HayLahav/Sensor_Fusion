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
