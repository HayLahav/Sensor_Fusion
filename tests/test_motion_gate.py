import numpy as np
import pytest

from fusion_perception.tracking.ego_motion import PoseEgoCompensation, interpolate_pose
from fusion_perception.tracking.motion_gate import (
    MOVING, STATIC, UNKNOWN, MotionObservation, classify_motion, ground_speed,
    warp_observations,
)


def _pose(x: float, yaw: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[0, 3] = x
    return T


# ── Pose interpolation ───────────────────────────────────────────────────────

def test_interpolate_pose_fills_gap():
    poses = {0: _pose(0.0, 0.0), 10: _pose(10.0, 0.2)}
    T = interpolate_pose(poses, 5)
    assert np.allclose(T[0, 3], 5.0)
    yaw = np.arctan2(T[1, 0], T[0, 0])
    assert yaw == pytest.approx(0.1, abs=1e-6)


def test_interpolate_pose_outside_range_is_none():
    poses = {5: _pose(0.0), 10: _pose(1.0)}
    assert interpolate_pose(poses, 2) is None
    assert interpolate_pose(poses, 11) is None


def test_ego_compensation_survives_missing_pose_frames():
    # Car drives 1 m/frame along IMU x; frames 3 and 4 are missing from poses.txt.
    poses = {f: _pose(float(f)) for f in range(10) if f not in (3, 4)}
    ego = PoseEgoCompensation(poses, np.eye(4), np.eye(3))
    assert ego.get_transform(0) is None          # first frame
    for f in range(1, 10):
        T = ego.get_transform(f)
        assert T is not None, f"lost compensation at frame {f}"
        # A static world point moves -1 m in x relative to the car every frame
        assert T[0, 3] == pytest.approx(-1.0, abs=1e-9)


def test_ego_compensation_handles_skipped_frames():
    poses = {f: _pose(float(f)) for f in range(20)}
    ego = PoseEgoCompensation(poses, np.eye(4), np.eye(3))
    ego.get_transform(0)
    T = ego.get_transform(5)                     # frame stride of 5
    assert T[0, 3] == pytest.approx(-5.0)


# ── Motion classification ────────────────────────────────────────────────────

def _obs(positions, source="lidar", start=0):
    return [MotionObservation(start + i, np.array(p, dtype=float), source)
            for i, p in enumerate(positions)]


def test_ground_speed_of_moving_object():
    obs = _obs([[0, 0, 10 + 0.5 * i] for i in range(6)])   # 0.5 m/frame
    assert ground_speed(obs, fps=10.0) == pytest.approx(5.0)


def test_classify_unknown_until_enough_observations():
    obs = _obs([[0, 0, 10]] * 3)
    assert classify_motion(obs, 10.0)[0] == UNKNOWN


def test_classify_static_parked_car_with_noise():
    rng = np.random.default_rng(0)
    pos = [[3 + rng.normal(0, 0.1), 0, 15 + rng.normal(0, 0.1)] for _ in range(6)]
    state, speed = classify_motion(_obs(pos), 10.0)
    assert state == STATIC and speed < 1.0


def test_classify_moving_car():
    state, _ = classify_motion(_obs([[0, 0, 10 + 0.8 * i] for i in range(6)]), 10.0)
    assert state == MOVING


def test_box_observations_need_more_frames():
    obs = _obs([[0, 0, 10]] * 6, source="box")
    assert classify_motion(obs, 10.0)[0] == UNKNOWN
    obs = _obs([[0, 0, 10]] * 10, source="box")
    assert classify_motion(obs, 10.0)[0] == STATIC


def test_warp_keeps_parked_car_in_place():
    # Car drives forward 1 m/frame (camera z). A parked car at z=20 world
    # appears at z=20-t in camera frame. Measurements are taken in each
    # frame's camera, and warping older ones by T_ego makes them agree.
    T_ego = np.eye(4)
    T_ego[2, 3] = -1.0
    obs: list[MotionObservation] = []
    for t in range(6):
        warp_observations(obs, T_ego)
        obs.append(MotionObservation(t, np.array([3.0, 0.0, 20.0 - t]), "lidar"))
    assert all(np.allclose(o.position, [3.0, 0.0, 15.0]) for o in obs)
    assert classify_motion(obs, 10.0)[0] == STATIC


# ── Tracker integration (needs torch + filterpy) ─────────────────────────────

def _det(frame_idx, z, x=3.0, lidar=None):
    from fusion_perception.utils.dataclasses import Detection3D
    return Detection3D(
        frame_idx=frame_idx, class_id=0, class_name="car", score=0.9,
        score_2d=0.9, score_3d=0.9, box_2d=[500, 150, 600, 220],
        box_3d=[x, 1.5, z, 1.8, 1.5, 4.2, 0.0],
        centroid_2d=[550, 185], centroid_3d=[x, 1.5, z], depth=z,
        lidar_centroid_3d=lidar,
    )


def test_tracker_keeps_new_parked_car_static_despite_noisy_first_box():
    pytest.importorskip("torch")
    pytest.importorskip("filterpy")
    from fusion_perception.tracking.kalman_cowtracker import KalmanCoWTracker

    tracker = KalmanCoWTracker(device="cpu", confirm_age=2)
    frame = np.zeros((375, 1242, 3), dtype=np.uint8)
    T_ego = np.eye(4)
    T_ego[2, 3] = -1.0                       # ego drives 1 m/frame forward
    # First box is 2 m too far (truncated / bad mono depth), then correct.
    zs = [22.0] + [20.0 - t for t in range(1, 8)]
    tracks = []
    for t, z in enumerate(zs):
        lidar = [3.0, 1.0, 20.0 - t]        # LiDAR median: steady
        tracks = tracker.update(frame, [_det(t, z, lidar=lidar)], t, fps=10.0,
                                T_ego=T_ego if t > 0 else None)
    assert len(tracks) == 1
    assert tracks[0].motion_state == STATIC
    assert np.allclose(tracks[0].velocity_3d, 0.0)
