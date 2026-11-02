"""Per-scan LiDAR labels from KITTI-360's accumulated semantic point clouds.

KITTI-360 does not label individual scans. It ships accumulated world-frame
clouds per frame window:
    data_3d_semantics/train/{seq}/static/{start:010d}_{end:010d}.ply
    data_3d_semantics/train/{seq}/dynamic/{start:010d}_{end:010d}.ply
with per-point 'semantic' (KITTI-360 label id) fields; dynamic clouds also
carry a 'timestamp' (frame index) per point.

To label a scan we move it into the world frame and take the label of the
nearest map point:
    T_world_velo = T_imu_to_world(frame) @ T_cam0_to_pose @ inv(T_cam0_to_velo)

The nearest-neighbour distances double as a calibration / pose check: with
correct extrinsics most scan points land within a few centimetres of the map.
A wrong or approximated transform (e.g. the velo≈pose fallback in
kitti_calibration.py) shows up as a large median distance.
Raw KITTI-360 scans are not motion-compensated (the sensor moves up to ~1 m
during one sweep at speed), so compare these numbers between calibrations or
frames rather than reading them as absolute calibration error.
"""
from __future__ import annotations
import os
import re
from glob import glob
from typing import Optional
import numpy as np

from fusion_perception.segmentation.lidar_classes import IGNORE, kitti360_to_train
from fusion_perception.tracking.ego_motion import interpolate_pose
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("kitti360_semantics")

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path: str) -> np.ndarray:
    """Minimal binary PLY reader (vertex element only) → structured array."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = None
        n_vertex = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: header has no end_header")
            tok = line.decode("ascii", "ignore").split()
            if not tok:
                continue
            if tok[0] == "format":
                fmt = tok[1]
            elif tok[0] == "element":
                in_vertex = tok[1] == "vertex"
                if in_vertex:
                    n_vertex = int(tok[2])
            elif tok[0] == "property" and in_vertex:
                if tok[1] == "list":
                    raise ValueError(f"{path}: list properties on vertices not supported")
                props.append((tok[2], _PLY_TYPES[tok[1]]))
            elif tok[0] == "end_header":
                break
        if fmt not in ("binary_little_endian", "binary_big_endian"):
            raise ValueError(f"{path}: unsupported PLY format {fmt}")
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + t) for name, t in props])
        return np.fromfile(f, dtype=dtype, count=n_vertex)


def load_cam_to_velo(calib_dir: str) -> np.ndarray:
    """calib_cam_to_velo.txt (3×4, one line) → 4×4 T_cam0_to_velo."""
    vals = np.loadtxt(os.path.join(calib_dir, "calib_cam_to_velo.txt")).reshape(-1)
    T = np.eye(4)
    T[:3, :] = vals[:12].reshape(3, 4)
    return T


class Kitti360ScanLabeler:
    """Label individual KITTI-360 scans by nearest-neighbour transfer."""

    def __init__(
        self,
        root: str,
        sequence: str,
        max_dist: float = 0.15,
        split: str = "train",
        include_dynamic: bool = True,
    ) -> None:
        from fusion_perception.data.kitti_calibration import _parse_cam_to_pose_360
        from fusion_perception.data.kitti360_labels import _load_poses

        self.root = root
        self.sequence = sequence
        self.max_dist = max_dist
        self.include_dynamic = include_dynamic
        calib_dir = os.path.join(root, "calibration")
        self.T_cam_to_pose = _parse_cam_to_pose_360(
            os.path.join(calib_dir, "calib_cam_to_pose.txt")).astype(np.float64)
        self.T_cam_to_velo = load_cam_to_velo(calib_dir)
        self.T_velo_to_pose = self.T_cam_to_pose @ np.linalg.inv(self.T_cam_to_velo)
        self.poses = _load_poses(os.path.join(root, "data_poses", sequence, "poses.txt"))

        sem_dir = os.path.join(root, "data_3d_semantics", split, sequence)
        self._static = self._index(os.path.join(sem_dir, "static"))
        self._dynamic = self._index(os.path.join(sem_dir, "dynamic"))
        if not self._static:
            raise FileNotFoundError(f"No semantic windows found under {sem_dir}/static")
        self._cache_key: Optional[tuple] = None
        self._cache = None
        logger.info(f"{sequence}: {len(self._static)} static windows, "
                    f"{len(self._dynamic)} dynamic windows")

    @staticmethod
    def _index(d: str) -> list[tuple[int, int, str]]:
        out = []
        for p in sorted(glob(os.path.join(d, "*.ply"))):
            m = re.match(r"(\d+)_(\d+)\.ply$", os.path.basename(p))
            if m:
                out.append((int(m.group(1)), int(m.group(2)), p))
        return out

    def frames_with_labels(self) -> list[int]:
        """Frames covered by a semantic window that also have a pose."""
        keys = sorted(self.poses)
        frames = set()
        for s, e, _ in self._static:
            frames.update(range(s, e + 1))
        return [f for f in sorted(frames) if keys and keys[0] <= f <= keys[-1]]

    def velo_to_world(self, frame_idx: int) -> Optional[np.ndarray]:
        P = interpolate_pose(self.poses, frame_idx)
        return None if P is None else P @ self.T_velo_to_pose

    def _window_for(self, frame_idx: int):
        for s, e, p in self._static:
            if s <= frame_idx <= e:
                return s, e, p
        return None

    def _load_window(self, frame_idx: int):
        from scipy.spatial import cKDTree

        win = self._window_for(frame_idx)
        if win is None:
            return None
        if self._cache_key == win[:2]:
            return self._cache
        s, e, p = win
        st = read_ply(p)
        xyz = np.stack([st["x"], st["y"], st["z"]], 1).astype(np.float64)
        sem = kitti360_to_train(st["semantic"])
        dyn = None
        if self.include_dynamic:
            for ds, de, dp in self._dynamic:
                if ds == s and de == e:
                    dyn = read_ply(dp)
                    break
        self._cache_key = (s, e)
        self._cache = (cKDTree(xyz), sem, dyn)
        return self._cache

    def label_scan(self, xyz_velo: np.ndarray, frame_idx: int) -> Optional[dict]:
        """Return {'labels', 'dist'} for each scan point, or None if unlabelled."""
        from scipy.spatial import cKDTree

        T = self.velo_to_world(frame_idx)
        win = self._load_window(frame_idx)
        if T is None or win is None:
            return None
        tree, sem, dyn = win
        world = xyz_velo[:, :3].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        dist, idx = tree.query(world, k=1, distance_upper_bound=1.0)
        found = np.isfinite(dist)
        labels = np.full(len(world), IGNORE, dtype=np.uint8)
        labels[found] = sem[idx[found]]

        # Moving objects: dynamic points recorded at this frame
        if dyn is not None and "timestamp" in dyn.dtype.names:
            d = dyn[dyn["timestamp"] == frame_idx]
            if len(d):
                dxyz = np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)
                ddist, didx = cKDTree(dxyz).query(world, k=1, distance_upper_bound=1.0)
                closer = np.isfinite(ddist) & (ddist < dist)
                labels[closer] = kitti360_to_train(d["semantic"])[didx[closer]]
                dist = np.where(closer, ddist, dist)

        labels[~(dist <= self.max_dist)] = IGNORE
        return {"labels": labels, "dist": dist}


def alignment_stats(dist: np.ndarray, xyz_velo: np.ndarray, max_range: float = 30.0) -> dict:
    """Summarise scan-to-map distances (calibration / pose quality check)."""
    near = np.linalg.norm(xyz_velo[:, :2], axis=1) <= max_range
    d = dist[near]
    d = np.where(np.isfinite(d), d, 1.0)
    if len(d) == 0:
        return {"n": 0}
    return {
        "n": int(len(d)),
        "median_m": float(np.median(d)),
        "p90_m": float(np.percentile(d, 90)),
        "frac_within_5cm": float((d <= 0.05).mean()),
        "frac_within_15cm": float((d <= 0.15).mean()),
    }
