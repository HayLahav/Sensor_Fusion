"""Build the LiDAR world map for a KITTI-360 drive and place it on a real map.

Outputs (in --out):
  lidar_map.ply            world-frame point cloud, coloured by height
  map_topdown.png          top-down render in the KITTI-360 world frame
With KITTI-360 GPS (OXTS, data_poses/<seq>/oxts/data) available, also:
  georef.json              world → latitude/longitude transform + fit error
  map_north_up.png         top-down render with north up (for web maps)
  paths.geojson            driven route (+ planned paths from planning.json)
  map_view.html            OpenStreetMap + LiDAR map + paths (open in a browser,
                           served from --out, e.g. `python -m http.server`)

Example:
  python scripts/build_lidar_map.py --kitti360-root /data/KITTI-360 \\
      --seq 2013_05_28_drive_0000_sync --start 0 --end 1000 --stride 2 \\
      --planning outputs/<run>/planning.json --out outputs/<run>/map
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_perception.data.kitti360_labels import _load_poses  # noqa: E402
from fusion_perception.data.kitti_calibration import Kitti360Calib  # noqa: E402
from fusion_perception.data.lidar_loader import Kitti360Lidar, load_bin  # noqa: E402
from fusion_perception.mapping.georef import (  # noqa: E402
    find_oxts_dir, fit_georeference, lines_geojson, load_oxts, write_leaflet_html,
)
from fusion_perception.mapping.lidar_map import (  # noqa: E402
    LidarWorldMap, height_colors, render_topdown, save_ply, velo_to_pose,
)
from fusion_perception.tracking.ego_motion import PoseEgoCompensation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kitti360-root", required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--stride", type=int, default=2, help="use every Nth scan")
    p.add_argument("--voxel", type=float, default=0.10)
    p.add_argument("--max-range", type=float, default=40.0)
    p.add_argument("--resolution", type=float, default=0.2, help="top-down render, metres per pixel")
    p.add_argument("--planning", default=None, help="planning.json from inference.py (optional)")
    p.add_argument("--plan-every", type=int, default=20, help="draw every Nth planned path")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _ego_path_to_world(path, T_cam_to_world):
    """Ego-frame path (x fwd, y left) → world points, at camera height."""
    p = np.asarray(path)[:, :2]
    cam = np.stack([-p[:, 1], np.zeros(len(p)), p[:, 0]], 1)
    return cam @ T_cam_to_world[:3, :3].T + T_cam_to_world[:3, 3]


def main() -> int:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib_dir = os.path.join(a.kitti360_root, "calibration")
    poses = _load_poses(os.path.join(a.kitti360_root, "data_poses", a.seq, "poses.txt"))
    lidar = Kitti360Lidar.from_dir(os.path.join(a.kitti360_root, "data_3d_raw", a.seq))

    world_map = LidarWorldMap(poses, velo_to_pose(calib_dir), voxel=a.voxel, max_range=a.max_range)
    end = a.end if a.end is not None else max(poses)
    for i, f in enumerate(range(a.start, end + 1, a.stride)):
        path = lidar.get_path(f)
        if path:
            world_map.add_scan(f, load_bin(path))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1} scans")
    pts = world_map.points()
    traj = world_map.trajectory()
    if not len(pts):
        print("No scans with poses in that range.")
        return 1
    print(f"Map: {len(pts):,} points from {len(world_map.frames_used)} scans")

    colors = height_colors(pts[:, 2])
    save_ply(str(out / "lidar_map.ply"), pts, colors)
    img, _ = render_topdown(pts[:, :2], pts[:, 2], a.resolution, traj[:, :2])
    cv2.imwrite(str(out / "map_topdown.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ── Georeference with KITTI-360 GPS ──────────────────────────────────────
    oxts_dir = find_oxts_dir(a.kitti360_root, a.seq)
    if oxts_dir is None:
        print("No OXTS GPS found (data_poses/<seq>/oxts/data) — skipping georeferencing.")
        return 0
    gps = load_oxts(oxts_dir)
    geo = fit_georeference({f: poses[f][:3, 3] for f in poses}, gps)
    geo.save(str(out / "georef.json"))
    print(f"Georeference: {geo.frames_used} frames, horizontal RMS {geo.rms_horizontal_m:.2f} m, "
          f"vertical RMS {geo.rms_vertical_m:.2f} m, scale {geo.fitted_scale:.4f} "
          f"({geo.frames_rejected} GPS outliers dropped)")

    enu = geo.world_to_enu(pts)
    traj_enu = geo.world_to_enu(traj)
    img_n, (e0, n0, e1, n1) = render_topdown(enu[:, :2], enu[:, 2], a.resolution)
    # Transparent where empty, so the street map shows through
    alpha = (img_n.sum(2) > 0).astype(np.uint8) * 255
    cv2.imwrite(str(out / "map_north_up.png"),
                np.dstack([cv2.cvtColor(img_n, cv2.COLOR_RGB2BGR), alpha]))
    from fusion_perception.mapping.georef import enu_to_lla
    lat_sw, lon_sw, _ = enu_to_lla(np.array([e0, n0, 0.0]), geo.lat0, geo.lon0, geo.alt0)
    lat_ne, lon_ne, _ = enu_to_lla(np.array([e1, n1, 0.0]), geo.lat0, geo.lon0, geo.alt0)
    bounds = [[float(lat_sw), float(lon_sw)], [float(lat_ne), float(lon_ne)]]

    lines = [{"lla": enu_to_lla(traj_enu, geo.lat0, geo.lon0, geo.alt0),
              "properties": {"name": "driven route", "color": "#00c853", "weight": 4}}]
    n_plans = 0
    if a.planning and os.path.exists(a.planning):
        calib = Kitti360Calib.from_dir(calib_dir)
        ego = PoseEgoCompensation(poses, calib.T_pose_to_cam, calib.R_rect)
        records = json.load(open(a.planning))["frames"]
        for r in records[:: max(1, a.plan_every)]:
            T = ego.rectcam_to_world(r["frame_idx"])
            pl = r.get("planner")
            if T is None or not pl or not pl.get("trajectories"):
                continue
            ok = pl["samples"][0]["check"]["ok"]
            w = _ego_path_to_world(pl["trajectories"][0], T)
            lines.append({"lla": geo.world_to_lla(w),
                          "properties": {"name": f"plan @ frame {r['frame_idx']}",
                                         "color": "#00b8d4" if ok else "#ff1744", "weight": 2,
                                         "reasoning": pl.get("reasoning")}})
            n_plans += 1
    gj = lines_geojson(lines)
    (out / "paths.geojson").write_text(json.dumps(gj))
    info = (f"{a.seq} · GPS fit RMS {geo.rms_horizontal_m:.2f} m · "
            f"green: driven route · cyan/red: planned paths ok/flagged ({n_plans})")
    write_leaflet_html(str(out / "map_view.html"), "map_north_up.png", bounds, gj, info)
    print(f"Wrote {out / 'map_view.html'} (serve the folder, e.g. `python -m http.server`, then open it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
