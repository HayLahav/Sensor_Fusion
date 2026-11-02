import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fusion_perception.data.kitti360_semantics import read_ply
from fusion_perception.mapping.georef import (
    enu_to_lla, fit_georeference, lla_to_enu,
)
from fusion_perception.mapping.lidar_map import (
    LidarWorldMap, LocalMapUnderlay, render_topdown, save_ply, voxel_downsample,
)
from fusion_perception.utils.dataclasses import OccupancyGrid, Track
from fusion_perception.visualization.bev_renderer import BEVOverlay, render_bev

REPO = Path(__file__).resolve().parents[1]
LAT0, LON0, ALT0 = 49.01, 8.42, 115.0           # Karlsruhe


# ── georeferencing ───────────────────────────────────────────────────────────

def test_lla_enu_roundtrip():
    enu = np.array([[1200.0, -800.0, 5.0], [0.0, 0.0, 0.0]])
    lat, lon, alt = enu_to_lla(enu, LAT0, LON0, ALT0)
    back = lla_to_enu(lat, lon, alt, LAT0, LON0, ALT0)
    assert np.allclose(back, enu, atol=1e-3)


def _world_to_enu_truth(world):
    a = np.radians(30.0)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return world @ R.T + np.array([350.0, -120.0, 2.0])


def test_fit_georeference_recovers_transform_and_rejects_glitch():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 300)
    world = np.c_[600 * t, 200 * np.sin(4 * t), 3 * t]     # curving 600 m drive
    enu = _world_to_enu_truth(world) + rng.normal(0, 0.05, world.shape)
    enu[150] += [40.0, 0, 0]                                # GPS glitch
    lat, lon, alt = enu_to_lla(enu, LAT0, LON0, ALT0)
    gps = {i: (lat[i], lon[i], alt[i]) for i in range(300)}
    geo = fit_georeference({i: world[i] for i in range(300)}, gps)
    assert geo.frames_rejected >= 1
    assert geo.rms_horizontal_m < 0.15
    assert geo.fitted_scale == pytest.approx(1.0, abs=1e-3)
    # A map point lands where the true transform puts it
    p = np.array([[100.0, 50.0, 1.0]])
    la, lo, _ = geo.world_to_lla(p)
    tla, tlo, _ = enu_to_lla(_world_to_enu_truth(p) - _world_to_enu_truth(world[:1])
                             + lla_to_enu(*gps[0], LAT0, LON0, ALT0)[None], LAT0, LON0, ALT0)
    e = lla_to_enu(la, lo, 0 * la, tla[0], tlo[0], 0)
    assert np.hypot(e[0, 0], e[0, 1]) < 0.2


def test_fit_needs_enough_frames():
    with pytest.raises(ValueError):
        fit_georeference({0: np.zeros(3)}, {0: (LAT0, LON0, ALT0)})


# ── LiDAR map ────────────────────────────────────────────────────────────────

def test_world_map_merges_static_point_seen_from_two_poses():
    poses = {0: np.eye(4), 1: np.eye(4)}
    poses[1][0, 3] = 5.0                                   # moved 5 m along world x
    m = LidarWorldMap(poses, np.eye(4), voxel=0.1)
    m.add_scan(0, np.array([[10.0, 0.0, 0.0, 0.5]]))
    m.add_scan(1, np.array([[5.0, 0.0, 0.0, 0.5]]))      # same world point
    pts = m.points()
    assert len(pts) == 1 and np.allclose(pts[0], [10.0, 0.0, 0.0], atol=0.05)
    assert m.trajectory().shape == (2, 3)


def test_voxel_downsample_and_ply(tmp_path):
    pts = np.array([[0.01, 0, 0], [0.02, 0, 0], [1.0, 1.0, 1.0]])
    ds, _ = voxel_downsample(pts, 0.1)
    assert len(ds) == 2
    save_ply(str(tmp_path / "m.ply"), ds, np.full((2, 3), 200, np.uint8))
    arr = read_ply(str(tmp_path / "m.ply"))
    assert len(arr) == 2 and set(arr.dtype.names) >= {"x", "red"}


def test_render_topdown_axes():
    xy = np.array([[0.0, 0.0], [10.0, 20.0]])
    img, (x0, y0, x1, y1) = render_topdown(xy, np.array([0.0, 1.0]), resolution=1.0, margin=0)
    assert img.shape[:2] == (21, 11)
    assert img[-1, 0].any() and img[0, -1].any()          # (0,0) bottom-left, (10,20) top-right


def test_local_underlay_moves_with_the_car():
    poses = {}
    for f in range(20):
        T = np.eye(4)
        T[2, 3] = 1.0 * f                                  # camera drives along its z
        poses[f] = T
    u = LocalMapUnderlay(lambda f: poses.get(f), voxel=0.1, every=1)
    wall = np.array([[2.0, 0.0, 30.0], [2.0, -1.0, 30.0]])  # above ground (y < 1.2)
    u.update(0, wall)
    xz = u.update(10, None)                                # nothing new seen
    assert np.allclose(np.sort(xz[:, 1]), [20.0, 20.0], atol=0.1)


# ── BEV rendering ────────────────────────────────────────────────────────────

def _grid(sem=True):
    rows, cols = 140, 100
    return OccupancyGrid(
        frame_idx=0, resolution=0.5, x_range=[-25.0, 25.0], z_range=[-10.0, 60.0],
        grid=[[0.0] * cols for _ in range(rows)], decay_factor=0.9,
        sem_grid=[[0] * cols for _ in range(rows)] if sem else None,
    )


def _track(state):
    return Track(track_id=3, class_name="car", first_seen=0, last_seen=0,
                 centroid_history=[[0, 0]], position_3d_history=[[5.0, 1.0, 20.0]],
                 cow_query_point=[0, 0], is_active=True, occlusion_count=0,
                 last_box_3d=[5.0, 1.0, 20.0, 1.8, 1.5, 4.2, 0.0],
                 velocity_3d=[0.0, 0.0, 8.0], motion_state=state)


@pytest.mark.parametrize("sem", [True, False])
def test_render_bev_with_overlay(sem):
    path = np.c_[np.linspace(1, 40, 40), np.zeros(40), np.zeros(40)]
    obstacle = np.zeros((140, 100), bool)
    obstacle[100, 50] = True
    ov = BEVOverlay(map_points_xz=np.array([[-20.0, 30.0]]), obstacle=obstacle,
                    driven_path=path, planner_paths=[path + [0, 3, 0]],
                    planner_bad=[{30, 31, 32}])
    img = render_bev(_grid(sem), [_track("static")], size_px=400, overlay=ov)
    assert img.shape == (400, 400, 3)
    px = img.reshape(-1, 3)
    assert ((px == [80, 220, 80]).all(1)).any()            # driven path (green)
    assert ((px == [255, 220, 0]).all(1)).any()            # planner path (cyan)
    assert ((px == [40, 40, 255]).all(1)).any()            # flagged waypoints (red)


def test_parked_track_has_no_velocity_arrow():
    moving = render_bev(_grid(), [_track("moving")], size_px=400)
    parked = render_bev(_grid(), [_track("static")], size_px=400)
    white = lambda im: int((im.reshape(-1, 3) == [255, 255, 255]).all(1).sum())
    assert white(moving) > white(parked)                   # arrow is drawn white


def test_render_bev_without_overlay_unchanged_size():
    assert render_bev(_grid(), [], size_px=300).shape == (300, 300, 3)


# ── build_lidar_map.py end to end on a tiny synthetic drive ──────────────────

def _fake_kitti360(root: Path, seq: str, n: int = 40):
    calib = root / "calibration"
    calib.mkdir(parents=True)
    # camera→pose: camera z (forward) = IMU x; camera x (right) = IMU y; camera y (down) = IMU z
    c2p = "0 0 1 0  1 0 0 0  0 1 0 0"
    (calib / "calib_cam_to_pose.txt").write_text(f"image_00: {c2p}\nimage_01: {c2p}\n")
    (calib / "calib_cam_to_velo.txt").write_text("0 0 1 0 -1 0 0 0 0 -1 0 0\n")
    (calib / "perspective.txt").write_text("R_rect_00: 1 0 0 0 1 0 0 0 1\n")
    pdir = root / "data_poses" / seq
    (pdir / "oxts" / "data").mkdir(parents=True)
    vdir = root / "data_3d_raw" / seq / "velodyne_points" / "data"
    vdir.mkdir(parents=True)
    world = []
    lines = []
    for f in range(n):
        x = 2.0 * f
        world.append([x, 0.0, 0.0])
        lines.append(f"{f} 1 0 0 {x} 0 1 0 0 0 0 1 0")
        # a wall 10 m to the side, seen from each pose (Velodyne: x fwd, y left)
        xs = np.arange(-20, 20, 0.5)
        pts = np.c_[xs, np.full_like(xs, 10.0), np.zeros_like(xs), np.ones_like(xs)]
        pts.astype(np.float32).tofile(vdir / f"{f:010d}.bin")
    (pdir / "poses.txt").write_text("\n".join(lines) + "\n")
    enu = _world_to_enu_truth(np.asarray(world))
    lat, lon, alt = enu_to_lla(enu, LAT0, LON0, ALT0)
    for f in range(n):
        (pdir / "oxts" / "data" / f"{f:010d}.txt").write_text(
            f"{lat[f]:.10f} {lon[f]:.10f} {alt[f]:.4f} 0 0 0\n")


def test_build_lidar_map_script(tmp_path):
    pytest.importorskip("scipy")
    _fake_kitti360(tmp_path / "K360", "seq")
    out = tmp_path / "map"
    env = {**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "build_lidar_map.py"),
                        "--kitti360-root", str(tmp_path / "K360"), "--seq", "seq",
                        "--stride", "1", "--out", str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    for name in ("lidar_map.ply", "map_topdown.png", "georef.json",
                 "map_north_up.png", "paths.geojson", "map_view.html"):
        assert (out / name).exists(), name
    geo = json.loads((out / "georef.json").read_text())
    assert geo["rms_horizontal_m"] < 0.05
    # The wall is one straight line in the world: all map points share one y
    ply = read_ply(str(out / "lidar_map.ply"))
    assert np.ptp(ply["y"]) < 0.2
    gj = json.loads((out / "paths.geojson").read_text())
    assert gj["features"][0]["geometry"]["type"] == "LineString"
    assert "leaflet@1.9.4" in (out / "map_view.html").read_text()


def test_plan_redrawn_in_current_ego_frame():
    sr = pytest.importorskip("fusion_perception.pipelines.stage_runner")
    T0 = np.eye(4)
    T5 = np.eye(4)
    T5[2, 3] = 5.0                                         # 5 m further along camera z
    path = np.c_[np.linspace(1, 20, 20), np.full(20, 2.0), np.zeros(20)]
    world = sr._ego_to_world(path, T0)
    again = sr._world_to_ego(world, T5)
    assert np.allclose(again[:, 0], path[:, 0] - 5.0) and np.allclose(again[:, 1], 2.0)
