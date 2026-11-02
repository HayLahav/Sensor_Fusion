"""Accumulated LiDAR world map for a KITTI-360 drive.

Each scan goes Velodyne → IMU/pose → world with KITTI-360's calibration:
    T_world_velo = T_imu_to_world(frame) @ T_cam0_to_pose @ inv(T_cam0_to_velo)

(The notebook used to chain the rectified camera transform with the
unrectified camera→pose one, and the camera→pose file was misread as
identity, which rotated every scan in the map.)

Points are voxel-downsampled as they are added, so a full drive fits in
memory without Open3D.
"""
from __future__ import annotations
import os
from typing import Optional
import cv2
import numpy as np

from fusion_perception.tracking.ego_motion import interpolate_pose


def velo_to_pose(calib_dir: str) -> np.ndarray:
    """4×4 Velodyne → IMU/pose transform from KITTI-360 calibration files."""
    from fusion_perception.data.kitti360_semantics import load_cam_to_velo
    from fusion_perception.data.kitti_calibration import _parse_cam_to_pose_360

    T_cam_to_pose = _parse_cam_to_pose_360(
        os.path.join(calib_dir, "calib_cam_to_pose.txt")).astype(np.float64)
    return T_cam_to_pose @ np.linalg.inv(load_cam_to_velo(calib_dir))


def voxel_downsample(pts: np.ndarray, voxel: float, extra: Optional[np.ndarray] = None):
    """Keep the mean point per voxel (and the mean of `extra` columns)."""
    if len(pts) == 0:
        return pts, extra
    q = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = np.zeros((len(counts), 3))
    np.add.at(out, inv, pts)
    out /= counts[:, None]
    ex = None
    if extra is not None:
        ex = np.zeros((len(counts), extra.shape[1]))
        np.add.at(ex, inv, extra)
        ex /= counts[:, None]
    return out, ex


class LidarWorldMap:
    """Accumulate world-frame LiDAR points for a drive.

    poses: {frame: 4×4 T_imu_to_world}; T_velo_to_pose from velo_to_pose().
    """

    def __init__(
        self,
        poses: dict[int, np.ndarray],
        T_velo_to_pose: np.ndarray,
        voxel: float = 0.10,
        max_range: float = 40.0,
        chunk_frames: int = 50,
    ) -> None:
        self.poses = poses
        self.T_velo_to_pose = np.asarray(T_velo_to_pose, dtype=np.float64)
        self.voxel = voxel
        self.max_range = max_range
        self.chunk_frames = chunk_frames
        self._chunks: list[np.ndarray] = []
        self._pending: list[np.ndarray] = []
        self.frames_used: list[int] = []

    def velo_to_world(self, frame_idx: int) -> Optional[np.ndarray]:
        P = interpolate_pose(self.poses, frame_idx)
        return None if P is None else P @ self.T_velo_to_pose

    def add_scan(self, frame_idx: int, pts_velo: np.ndarray) -> bool:
        T = self.velo_to_world(frame_idx)
        if T is None:
            return False
        p = pts_velo[:, :3].astype(np.float64)
        p = p[np.linalg.norm(p[:, :2], axis=1) <= self.max_range]
        self._pending.append(p @ T[:3, :3].T + T[:3, 3])
        self.frames_used.append(frame_idx)
        if len(self._pending) >= self.chunk_frames:
            self._flush()
        return True

    def _flush(self) -> None:
        if self._pending:
            pts, _ = voxel_downsample(np.concatenate(self._pending), self.voxel)
            self._chunks.append(pts)
            self._pending = []

    def points(self) -> np.ndarray:
        self._flush()
        if not self._chunks:
            return np.zeros((0, 3))
        pts, _ = voxel_downsample(np.concatenate(self._chunks), self.voxel)
        self._chunks = [pts]
        return pts

    def trajectory(self) -> np.ndarray:
        """[N,3] world positions of the IMU for the frames used."""
        out = [interpolate_pose(self.poses, f)[:3, 3] for f in self.frames_used]
        return np.asarray(out) if out else np.zeros((0, 3))


def height_colors(z: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    """[N] heights → [N,3] uint8 RGB (plasma-like, no matplotlib needed)."""
    lo = np.percentile(z, 2) if lo is None else lo
    hi = np.percentile(z, 98) if hi is None else hi
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_PLASMA)[:, 0, ::-1]
    return lut[(t * 255).astype(np.uint8)]


def save_ply(path: str, pts: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """Binary little-endian PLY (x, y, z float32 + optional uchar RGB)."""
    n = len(pts)
    props = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        props += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    arr = np.zeros(n, dtype=props)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    if colors is not None:
        arr["red"], arr["green"], arr["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property {'float' if t == '<f4' else 'uchar'} {name}" for name, t in props]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(arr.tobytes())


def render_topdown(
    xy: np.ndarray,
    z: np.ndarray,
    resolution: float = 0.2,
    trajectory_xy: Optional[np.ndarray] = None,
    margin: float = 5.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Top-down RGB raster of points (highest point per pixel wins).

    Axes: image right = +x, image up = +y. Returns (image, (xmin, ymin, xmax, ymax)).
    """
    xmin, ymin = xy.min(axis=0) - margin
    xmax, ymax = xy.max(axis=0) + margin
    w = int(np.ceil((xmax - xmin) / resolution)) + 1
    h = int(np.ceil((ymax - ymin) / resolution)) + 1
    img = np.zeros((h, w, 3), np.uint8)
    col = ((xy[:, 0] - xmin) / resolution).astype(np.intp)
    row = (h - 1 - (xy[:, 1] - ymin) / resolution).astype(np.intp)
    order = np.argsort(z)                     # draw low first, high last
    img[row[order], col[order]] = height_colors(z)[order]
    if trajectory_xy is not None and len(trajectory_xy) > 1:
        tc = ((trajectory_xy[:, 0] - xmin) / resolution).astype(np.int32)
        tr = (h - 1 - (trajectory_xy[:, 1] - ymin) / resolution).astype(np.int32)
        cv2.polylines(img, [np.stack([tc, tr], 1)], False, (0, 255, 0), 2, cv2.LINE_AA)
    return img, (float(xmin), float(ymin), float(xmax), float(ymax))


class LocalMapUnderlay:
    """Rolling LiDAR map around the car for the BEV underlay.

    Keeps world-frame points from recent frames (voxel-downsampled) and
    returns them in the current camera frame each frame.
    rectcam_to_world: callable(frame_idx) → 4×4 or None
    (PoseEgoCompensation.rectcam_to_world).
    """

    def __init__(self, rectcam_to_world, voxel: float = 0.25, radius: float = 70.0,
                 every: int = 2, max_points: int = 400_000) -> None:
        self.rectcam_to_world = rectcam_to_world
        self.voxel = voxel
        self.radius = radius
        self.every = max(1, every)
        self.max_points = max_points
        self._world = np.zeros((0, 3))

    def update(self, frame_idx: int, lidar_pts_cam: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Add this frame's points; return [N,2] (x, z) of the map in the current camera frame."""
        T = self.rectcam_to_world(frame_idx)
        if T is None:
            return None
        if lidar_pts_cam is not None and len(lidar_pts_cam) and frame_idx % self.every == 0:
            p = lidar_pts_cam[:, :3].astype(np.float64)
            # Skip the ground (camera y points down; ground ≈ +1.5 m) to keep obstacles
            p = p[p[:, 1] < 1.2]
            w = p @ T[:3, :3].T + T[:3, 3]
            pts = np.concatenate([self._world, w])
            centre = T[:3, 3]
            pts = pts[np.linalg.norm(pts - centre, axis=1) <= self.radius]
            self._world, _ = voxel_downsample(pts, self.voxel)
            if len(self._world) > self.max_points:
                self._world = self._world[-self.max_points:]
        if not len(self._world):
            return None
        Tinv = np.linalg.inv(T)
        cam = self._world @ Tinv[:3, :3].T + Tinv[:3, 3]
        return cam[:, [0, 2]]
