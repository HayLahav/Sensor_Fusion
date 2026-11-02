import json

import cv2
import numpy as np
import pytest

from fusion_perception.data.kitti_calibration import _parse_cam_to_pose_360
from fusion_perception.planning.audit import PlanningAudit, PredictionsFileSource
from fusion_perception.planning.ego_state import EgoStateExtractor
from fusion_perception.planning.kitti360_views import (
    FRONT, FRONT_LEFT, FRONT_RIGHT, VIEW_TAGS, Kitti360ViewRenderer, MEICamera,
)
from fusion_perception.planning.qwen_drive_scene import (
    camera_frame_offsets, frame_from_token, instruction_text, read_predictions,
    scene_record, scene_token,
)
from fusion_perception.tracking.ego_motion import PoseEgoCompensation

# KITTI-360 calib_cam_to_pose.txt layout: name and 12 values on one line
CAM0 = ("0.0371783278 -0.0986182135 0.9944306009 1.5752681039 0.9992675562 -0.0053553387 "
        "-0.0378902567 0.0043914093 0.0090621821 0.9951109327 0.0983468786 -0.6500000000")
# Idealised fisheyes in the same IMU frame as CAM0 (x forward, y right,
# z down): image_02 looks left with image-right pointing forward, image_03
# looks right with image-right pointing backward.
CAM2 = "1 0 0 0.5 0 0 -1 -0.5 0 1 0 -0.5"
CAM3 = "-1 0 0 0.5 0 0 1 0.5 0 1 0 -0.5"


def _write_calib(d, cam0=CAM0):
    c = d / "calibration"
    c.mkdir(parents=True, exist_ok=True)
    (c / "calib_cam_to_pose.txt").write_text(
        f"image_00: {cam0}\nimage_01: {cam0}\nimage_02: {CAM2}\nimage_03: {CAM3}\n")
    for cam in ("image_02", "image_03"):
        (c / f"{cam}.yaml").write_text(
            "%YAML:1.0\n---\nmodel_type: MEI\ncamera_name: " + cam + "\n"
            "image_width: 400\nimage_height: 400\nmirror_parameters:\n   xi: 1.0\n"
            "distortion_parameters:\n   k1: 0.0\n   k2: 0.0\n   p1: 0.0\n   p2: 0.0\n"
            "projection_parameters:\n   gamma1: 200.0\n   gamma2: 200.0\n"
            "   u0: 200.0\n   v0: 200.0\n")
    return c


# ── calibration parsing (ego-motion bug) ─────────────────────────────────────

def test_cam_to_pose_single_line_format(tmp_path):
    c = _write_calib(tmp_path)
    T = _parse_cam_to_pose_360(str(c / "calib_cam_to_pose.txt"))
    assert T[0, 3] == pytest.approx(1.5752681)
    # camera forward (z) is IMU forward (x)
    assert T[0, 2] == pytest.approx(0.9944306)


def test_cam_to_pose_multiline_and_missing(tmp_path):
    vals = CAM0.split()
    p = tmp_path / "c.txt"
    p.write_text("image_00:\n" + " ".join(vals[:4]) + "\n" + " ".join(vals[4:8]) + "\n"
                 + " ".join(vals[8:]) + "\n")
    assert _parse_cam_to_pose_360(str(p))[2, 3] == pytest.approx(-0.65)
    with pytest.raises(ValueError):
        _parse_cam_to_pose_360(str(p), "image_03")


def test_forward_driving_compensates_along_camera_depth(tmp_path):
    """IMU moves 1 m forward (its x). A static point must come 1 m closer in
    camera depth (z), not shift sideways — what the identity fallback did."""
    c = _write_calib(tmp_path)
    T_cam_to_pose = _parse_cam_to_pose_360(str(c / "calib_cam_to_pose.txt")).astype(np.float64)
    poses = {0: np.eye(4), 1: np.eye(4)}
    poses[1][0, 3] = 1.0
    ego = PoseEgoCompensation(poses, np.linalg.inv(T_cam_to_pose), np.eye(3))
    ego.get_transform(0)
    t = ego.get_transform(1)[:3, 3]
    assert t[2] == pytest.approx(-0.994, abs=0.01)
    assert abs(t[0]) < 0.12


# ── fisheye views ────────────────────────────────────────────────────────────

def test_mei_projects_optical_axis_to_principal_point(tmp_path):
    cam = MEICamera.from_yaml(str(_write_calib(tmp_path) / "image_02.yaml"))
    u, v = cam.project(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]))
    assert (u[0], v[0]) == pytest.approx((200.0, 200.0))
    assert np.isnan(u[1])                         # straight behind: not visible


def _write_frames(root, seq, frames):
    for f in frames:
        d0 = root / "data_2d_raw" / seq / "image_00" / "data_rect"
        d0.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d0 / f"{f:010d}.png"), np.full((376, 1408, 3), 50, np.uint8))
        for cam in ("image_02", "image_03"):
            d = root / "data_2d_raw" / seq / cam / "data_rgb"
            d.mkdir(parents=True, exist_ok=True)
            img = np.zeros((400, 400, 3), np.uint8)
            img[:, 200:] = 255                    # right half of each fisheye image bright
            cv2.imwrite(str(d / f"{f:010d}.png"), img)


def test_fisheye_side_views_render(tmp_path):
    _write_calib(tmp_path)
    _write_frames(tmp_path, "seq", [0])
    r = Kitti360ViewRenderer(str(tmp_path), "seq", side_mode="fisheye",
                             side_yaw_deg=55, side_hfov_deg=60, side_size=(120, 80))
    views = r.render(0)
    assert set(views) == set(VIEW_TAGS)
    assert views[FRONT_LEFT].shape == (80, 120, 3)
    # Left view turned 35° forward of the left fisheye's axis: the fisheye's
    # image-right is forward, so the virtual view sees mostly its bright half
    assert views[FRONT_LEFT].mean() > 150
    # Right fisheye's image-right is backward: front-right view sees the dark half
    assert views[FRONT_RIGHT].mean() < 100


def test_front_crop_and_missing_frames(tmp_path):
    _write_frames(tmp_path, "seq", [0])
    r = Kitti360ViewRenderer(str(tmp_path), "seq", side_mode="front_crop")
    v = r.render(0)
    assert v[FRONT_LEFT].shape[1] == 704
    assert r.render(5) is None
    assert r.render_history(0, (-5, 0)) is None


# ── scene file + predictions ─────────────────────────────────────────────────

def _straight_state(frame=30):
    poses = {f: np.eye(4) for f in range(100)}
    for f in poses:
        poses[f][2, 3] = 1.0 * f                  # camera forward 10 m/s at 10 Hz
    return EgoStateExtractor(PoseEgoCompensation(poses, np.eye(4), np.eye(3))), poses


def test_scene_record_layout():
    ex, _ = _straight_state()
    state = ex.get(30)
    offs = camera_frame_offsets(10.0)
    assert offs == (-15, -10, -5, 0)
    paths = {t: [f"frames/{i}/{30 + k:010d}.jpg" for k in offs] for i, t in enumerate(VIEW_TAGS)}
    rec = scene_record("seq", state, paths)
    content = rec["messages"][0]["content"]
    images = [c for c in content if "image" in c]
    assert len(images) == 12 and images[0]["image"].startswith("frames/0/")
    assert content[0] == {"text": FRONT}
    assert content[-1]["text"].endswith("[GO STRAIGHT]")
    assert " -t-1.5s: (-15.0000, 0.0000, 0.0000);" in content[-1]["text"]
    t = rec["trajectory"]
    assert np.asarray(t["hist_traj_10hz"]).shape == (16, 3)
    assert np.asarray(t["hist_vel_10hz"]).shape == (16, 2)
    assert np.asarray(t["future_traj_10hz"]).shape == (50, 3)
    assert t["ego_status"]["driving_command"] == [0, 1, 0, 0]
    assert rec["meta_info"]["token"] == scene_token("seq", 30) == "seq_0000000030"
    assert frame_from_token("seq_0000000030") == 30
    json.dumps(rec)                                # serialisable


def test_instruction_matches_qwen_drive_format():
    hist = np.zeros((16, 3))
    txt = instruction_text(hist, 1)
    assert txt.startswith("The input images are organized by camera view. Each view contains 4 "
                          "temporal frames captured at 0.5s intervals")
    assert txt.count(" -t-") == 4 and txt.endswith("[TURN LEFT]")


def test_predictions_drive_planner_audit(tmp_path):
    ex, _ = _straight_state()
    driven = np.asarray(ex.get(30).future)
    bad = driven.copy()
    bad[:, 1] = -8.0                              # 8 m to the right
    pred = tmp_path / "predictions.jsonl"
    with open(pred, "w") as f:
        f.write(json.dumps({"token": "seq_0000000030", "trajectories": [driven.tolist(), bad.tolist()],
                            "reasoning": "Road is clear, keep lane."}) + "\n")
        f.write('{"token": "seq_00000000')        # truncated last line is ignored
    assert list(read_predictions(str(pred))) == ["seq_0000000030"]

    from fusion_perception.utils.dataclasses import Track
    wall = Track(track_id=1, class_name="car", first_seen=0, last_seen=0,
                 centroid_history=[[0, 0]], position_3d_history=[[8.0, 1.0, 10.0]],
                 cow_query_point=[0, 0], is_active=True, occlusion_count=0,
                 last_box_3d=[8.0, 1.0, 10.0, 2.0, 1.5, 6.0, 0.0], motion_state="static")
    audit = PlanningAudit(ex, plan_source=PredictionsFileSource(str(pred), "seq"),
                          every=1000)
    grid = dict(resolution=0.5, x_range=[-25.0, 25.0], z_range=[-10.0, 60.0])
    from fusion_perception.planning.bev_check import build_plan_context  # noqa: F401
    rec = audit.update(30, {"tracks": [wall]}, None)
    assert rec is not None and rec["planner"]["reasoning"].startswith("Road")
    first, second = rec["planner"]["samples"]
    assert first["check"]["ok"] and first["metrics"]["ADE@3s"] == pytest.approx(0.0)
    assert not second["check"]["ok"] and second["metrics"]["FDE@3s"] == pytest.approx(8.0)
    assert audit.update(31, {"tracks": []}, None) is None   # no plan, not an audit frame
    s = audit.summary()["planner"]
    assert s["frames"] == 1 and s["flag_rate"] == 0.0
    assert s["metrics_best_sample"]["min_ADE@3s"] == pytest.approx(0.0)
