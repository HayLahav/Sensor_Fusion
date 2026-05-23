import numpy as np
from fusion_perception.tracking.association import (
    build_cost_matrix, mahalanobis_gate, hungarian_match
)

def _box(cx=0., cz=10., l=4., w=2., h=1.5, theta=0., cy=0.):
    return [cx, cy, cz, theta, l, w, h]

def test_cost_matrix_shape():
    pred = [_box(0.), _box(5.)]
    dets = [_box(0.1), _box(5.1), _box(20.)]
    C = build_cost_matrix(pred, dets, cow_valid={0, 1}, alpha=0.35)
    assert C.shape == (2, 3)

def test_cost_perfect_match_is_low():
    box = _box(0., 10.)
    C = build_cost_matrix([box], [box], cow_valid={0}, alpha=0.35)
    assert C[0, 0] < 0.1

def test_cost_no_overlap_is_high():
    a = _box(0., 10.)
    b = _box(100., 10.)
    C = build_cost_matrix([a], [b], cow_valid={0}, alpha=0.35)
    assert C[0, 0] > 0.8

def test_cost_cow_fallback_matches_iou_only():
    """When CoW fails (empty set), cost == alpha*iou_cost + (1-alpha)*iou_cost == iou_cost."""
    a = _box(0., 10.)
    b = _box(100., 10.)
    C_cow = build_cost_matrix([a], [b], cow_valid=set(), alpha=0.35)
    iou_cost = 1.0  # no overlap
    assert abs(C_cow[0, 0] - iou_cost) < 1e-9

def test_mahalanobis_gate_removes_distant():
    pred_states = [np.array([0., 0., 10., 0., 4., 2., 1.5, 0., 0., 0.]),
                   np.array([50., 0., 10., 0., 4., 2., 1.5, 0., 0., 0.])]
    pred_covs = [np.eye(10) * 0.1, np.eye(10) * 0.1]
    det_z = [np.array([0.1, 0., 10.1, 0., 4., 2., 1.5]),
             np.array([0.1, 0., 10.1, 0., 4., 2., 1.5])]
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
