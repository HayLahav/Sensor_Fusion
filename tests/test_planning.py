import numpy as np
import pytest

from fusion_perception.planning.audit import DrivenPathAudit
from fusion_perception.planning.bev_check import (
    BEVCheckConfig, build_plan_context, check_trajectory,
)
from fusion_perception.planning.ego_state import EgoStateExtractor
from fusion_perception.planning.metrics import displacement_errors
from fusion_perception.tracking.ego_motion import PoseEgoCompensation
from fusion_perception.utils.dataclasses import Track

GRID = dict(resolution=0.5, x_range=[-25.0, 25.0], z_range=[-10.0, 60.0])
CAM_H = 1.6     # camera height above flat ground (camera y points down)


# ── helpers ──────────────────────────────────────────────────────────────────

def _cam_pose(x, z, yaw):
    """Camera→world pose. World uses camera axes; yaw > 0 turns left (towards -x)."""
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, 0] = [c, 0, s]
    T[:3, 1] = [0, 1, 0]
    T[:3, 2] = [-s, 0, c]
    T[:3, 3] = [x, 0, z]
    return T


def _extractor(poses, fps=10.0):
    # identity calibration → poses are camera→world directly
    return EgoStateExtractor(PoseEgoCompensation(poses, np.eye(4), np.eye(3)), fps=fps)


def _straight_poses(n=120, v=10.0, fps=10.0):
    return {f: _cam_pose(0.0, v * f / fps, 0.0) for f in range(n)}


def _left_turn_poses(n=120, v=8.0, radius=20.0, fps=10.0):
    poses = {}
    for f in range(n):
        yaw = v * f / fps / radius
        # circle turning towards -x
        poses[f] = _cam_pose(-radius * (1 - np.cos(yaw)), radius * np.sin(yaw), yaw)
    return poses


def _ground(step=0.4):
    xs, zs = np.meshgrid(np.arange(-24, 24, step), np.arange(-9, 59, step))
    return np.c_[xs.ravel(), np.full(xs.size, CAM_H), zs.ravel()]


def _box_points(x0, x1, z0, z1, h=1.5, step=0.2):
    xs, ys, zs = np.meshgrid(np.arange(x0, x1, step), np.arange(CAM_H - h, CAM_H - 0.2, step),
                             np.arange(z0, z1, step))
    return np.c_[xs.ravel(), ys.ravel(), zs.ravel()]


def _straight_path(dist=30.0, n=50):
    x = np.linspace(dist / n, dist, n)
    return np.c_[x, np.zeros(n), np.zeros(n)]


def _track(tid, x, z, vx=0.0, vz=0.0, state="static", cls="car"):
    return Track(
        track_id=tid, class_name=cls, first_seen=0, last_seen=0,
        centroid_history=[[0, 0]], position_3d_history=[[x, 1.0, z]],
        cow_query_point=[0, 0], is_active=True, occlusion_count=0,
        last_box_3d=[x, 1.0, z, 1.8, 1.5, 4.2, 0.0],
        velocity_3d=[vx, 0.0, vz], motion_state=state,
    )


# ── ego state ────────────────────────────────────────────────────────────────

def test_ego_state_straight():
    s = _extractor(_straight_poses()).get(30)
    assert s.velocity == pytest.approx([10.0, 0.0], abs=1e-6)
    assert s.acceleration == pytest.approx([0.0, 0.0], abs=1e-6)
    assert len(s.history) == 16 and s.history[-1] == pytest.approx([0, 0, 0])
    assert s.history[0][0] == pytest.approx(-15.0)
    assert len(s.future) == 50 and s.future[0][0] == pytest.approx(1.0)
    assert s.command == "straight"
    q = s.to_qwen_drive()
    assert q["ego_status"]["driving_command"] == [0, 1, 0, 0]
    assert len(q["expert_ego_vector"]) == 8


def test_ego_state_left_turn_is_positive_y_and_heading():
    s = _extractor(_left_turn_poses()).get(20)
    end = s.future[-1]
    assert end[1] > 0 and end[2] > 0          # left of start, heading turned left
    assert s.command == "left"
    assert s.speed == pytest.approx(8.0, rel=0.02)


def test_ego_state_unknown_at_end_and_none_without_history():
    ex = _extractor(_straight_poses(n=60))
    assert ex.get(5) is None                   # less than 1.5 s of past
    s = ex.get(59)                             # last frame: nothing ahead
    assert s.future is None and s.command == "unknown"
    s = ex.get(57)                             # two real future poses, rest padded
    assert sum(s.future_valid) == 2 and len(s.future) == 50
    assert len(s.valid_future()) == 2


def test_history_dynamics_in_body_frame():
    s = _extractor(_left_turn_poses()).get(30)
    hv = np.asarray(s.history_velocity)
    assert hv.shape == (16, 2) and np.asarray(s.history_acceleration).shape == (16, 2)
    # Constant-speed circle: in each point's own frame the car moves straight ahead
    assert np.allclose(hv[:, 0], 8.0, rtol=0.02) and np.allclose(hv[:, 1], 0.0, atol=0.1)
    # Centripetal acceleration v²/r = 3.2 m/s² to the left
    assert np.asarray(s.history_acceleration)[-1, 1] == pytest.approx(3.2, rel=0.05)


def test_ego_state_fills_missing_pose_frames():
    poses = {f: p for f, p in _straight_poses().items() if f not in (28, 29, 31)}
    s = _extractor(poses).get(30)
    assert s.velocity[0] == pytest.approx(10.0, abs=1e-6)


# ── BEV check ────────────────────────────────────────────────────────────────

def test_flat_ground_is_not_an_obstacle():
    ctx = build_plan_context([], lidar_pts_cam=_ground(), **GRID)
    assert not ctx.obstacle.any()
    assert check_trajectory(_straight_path(), ctx).ok


def test_obstacle_ahead_is_flagged_and_swerve_avoids_it():
    pts = np.vstack([_ground(), _box_points(-1.0, 1.0, 20.0, 21.0)])
    ctx = build_plan_context([], lidar_pts_cam=pts, **GRID)
    res = check_trajectory(_straight_path(), ctx)
    assert not res.ok and {v.kind for v in res.violations} == {"obstacle"}
    # Same distance but 5 m to the left (y > 0)
    path = _straight_path()
    path[:, 1] = np.clip((path[:, 0] - 5) / 10, 0, 1) * 5.0
    assert check_trajectory(path, ctx).ok


def test_semantic_labels_remove_ground_and_mark_off_road():
    ground = _ground()
    obstacle = _box_points(-1.0, 1.0, 20.0, 21.0)
    pts = np.vstack([ground, obstacle])
    # ground: road for x < 3 (camera x right), sidewalk beyond; obstacle: car
    labels = np.r_[np.where(ground[:, 0] < 3.0, 0, 1), np.full(len(obstacle), 8)].astype(np.uint8)
    ctx = build_plan_context([], lidar_pts_cam=pts, lidar_labels=labels, **GRID)
    res = check_trajectory(_straight_path(dist=15.0, n=15), ctx)
    assert res.ok                              # stops before the car, stays on road
    right = _straight_path(dist=15.0, n=15)
    right[:, 1] = -6.0                         # 6 m to the right → x_cam = +6 (sidewalk)
    kinds = {v.kind for v in check_trajectory(right, ctx).violations}
    assert "off_road" in kinds


def test_moving_agent_conflict_and_parked_car_obstacle():
    # Oncoming car in our lane, 30 m ahead at 10 m/s → meets us around 1.5 s
    moving = _track(1, 0.0, 30.0, vz=-10.0, state="moving")
    ctx = build_plan_context([moving], **GRID)
    res = check_trajectory(_straight_path(), ctx)
    assert any(v.kind == "agent" for v in res.violations)
    assert not ctx.obstacle.any()              # moving tracks aren't static obstacles

    parked = _track(2, 0.0, 30.0, state="static")
    ctx = build_plan_context([parked], **GRID)
    res = check_trajectory(_straight_path(), ctx)
    assert {v.kind for v in res.violations} == {"obstacle"}

    # Same oncoming car in the next lane (3.5 m to the left) → no conflict
    other_lane = _track(3, -3.5, 30.0, vz=-10.0, state="moving")
    res = check_trajectory(_straight_path(), build_plan_context([other_lane], **GRID))
    assert res.ok and res.min_agent_clearance > 0.5


def test_bad_trajectory_shape_raises():
    ctx = build_plan_context([], **GRID)
    with pytest.raises(ValueError):
        check_trajectory(np.zeros((5, 2)), ctx)


# ── metrics + audit ──────────────────────────────────────────────────────────

def test_displacement_errors():
    gt = _straight_path(dist=50.0)             # 1 m per 0.1 s
    pred = gt.copy()
    pred[:, 1] = 0.5
    m = displacement_errors(pred, gt)
    assert m["ADE@1s"] == pytest.approx(0.5) and m["FDE@3s"] == pytest.approx(0.5)


def test_driven_path_audit_counts_false_alarms():
    ex = _extractor(_straight_poses())
    audit = DrivenPathAudit(ex, BEVCheckConfig())
    from fusion_perception.utils.dataclasses import OccupancyGrid

    def occ(f):
        rows, cols = 140, 100
        return OccupancyGrid(frame_idx=f, grid=[[0.0] * cols for _ in range(rows)],
                             decay_factor=0.9, **GRID)

    for f in range(20, 30):
        audit.update(f, {"tracks": [], "occupancy": occ(f)}, _ground())
    # A parked car right in our path on one frame → one flagged frame
    audit.update(30, {"tracks": [_track(1, 0.0, 15.0)], "occupancy": occ(30)}, _ground())
    s = audit.summary()
    assert s["frames_checked"] == 11
    assert s["flagged_frames"] == [30]
    assert s["false_alarm_rate"] == pytest.approx(1 / 11)
    assert s["commands"] == {"straight": 11}


def test_sloped_ground_is_not_an_obstacle():
    g = _ground()
    g[:, 1] -= 0.06 * g[:, 2]                  # 6% uphill road ahead
    ctx = build_plan_context([], lidar_pts_cam=g, **GRID)
    assert not ctx.obstacle.any()
    assert check_trajectory(_straight_path(), ctx).ok
