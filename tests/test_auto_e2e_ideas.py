import numpy as np
import pytest
from fusion_perception.utils.projection import PinholeProjection, FisheyeProjection, as_projection
from fusion_perception.evaluation.metrics import compute_trajectory_metrics, trajectory_gate
from fusion_perception.tracking.ego_motion import EgoMotionHistory
from fusion_perception.occupancy.bev_grid import sem_grid_to_channels, OccupancyBEVGenerator
from fusion_perception.visualization.box3d_projector import project_box3d

K = np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1]], dtype=np.float32)


def test_pinhole_matches_matrix_and_rejects_behind():
    uv, ok = PinholeProjection(K).project(np.array([[1.0, 0.5, 10.0], [0, 0, -1.0]]))
    assert np.allclose(uv[0], [390, 275]) and ok.tolist() == [True, False]


def test_fisheye_center_and_bad_K():
    uv, ok = FisheyeProjection(K).project(np.array([[0.0, 0.0, 5.0]]))
    assert np.allclose(uv[0], [320, 240]) and ok[0]
    with pytest.raises(ValueError):
        PinholeProjection(np.eye(4))
    assert isinstance(as_projection(K), PinholeProjection)


def test_project_box3d_accepts_matrix_and_projection():
    box = [0, 0, 10, 1.5, 4, 2]
    a = np.zeros((480, 640, 3), np.uint8); b = a.copy()
    project_box3d(a, box, K, (255, 0, 0)); project_box3d(b, box, PinholeProjection(K), (255, 0, 0))
    assert a.any() and (a == b).all()


def test_ade_fde_and_gate():
    m = compute_trajectory_metrics({1: [[0, 0], [1, 0]], 2: [[0, 0]]}, {1: [[0, 0], [1, 2]], 3: [[0, 0]]})
    assert m["n"] == 1 and m["ade"] == pytest.approx(1.0) and m["fde"] == pytest.approx(2.0)
    assert trajectory_gate(m) and not trajectory_gate(m, max_fde=1.0)
    assert not trajectory_gate(compute_trajectory_metrics({}, {}))


def test_ego_history_straight_drive():
    h = EgoMotionHistory(steps=4, dt=0.1)
    T = np.eye(4); T[2, 3] = -1.0  # moved 1 m forward -> old points shift -1 in z
    for _ in range(2):
        h.update(T)
    arr = h.as_array()
    assert arr.shape == (4, 4) and not arr[:2].any()
    assert arr[-1, 0] == pytest.approx(10.0) and arr[-1, 1] == pytest.approx(0.0)
    assert arr[-1, 2] == pytest.approx(0.0, abs=1e-6)


def test_semantic_channels():
    g = np.array([[13, -1], [0, 13]])
    ch = sem_grid_to_channels(g, (0, 13))
    assert ch.shape == (2, 2, 2) and ch[1].sum() == 2 and ch[0].sum() == 1
    gen = OccupancyBEVGenerator(1.0, [-2, 2], [0, 4], 0.9)
    assert gen.get_semantic_channels().shape[1:] == (4, 4)
