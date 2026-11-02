"""Front, front-left and front-right camera views from KITTI-360.

Qwen-Drive was trained on three forward cameras (front, front-left,
front-right). KITTI-360 has a forward perspective camera (image_00) and two
side-facing fisheye cameras (image_02 left, image_03 right). The side views
are made by rendering a virtual pinhole camera, turned `side_yaw_deg` from
straight ahead, out of each fisheye image (MEI model, calibration in
calibration/image_02.yaml / image_03.yaml).

side_mode:
  "fisheye"    virtual pinhole views from the fisheyes (closest to training)
  "front_crop" left / right parts of the front image (no extra field of view;
               use when fisheye data isn't downloaded)
  "blank"      grey images (lets the planner rely on the front view only)
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("kitti360_views")

FRONT, FRONT_LEFT, FRONT_RIGHT = "<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>"
VIEW_TAGS = (FRONT, FRONT_LEFT, FRONT_RIGHT)   # Qwen-Drive's order (scene.CAMERA_VIEWS)


@dataclass
class MEICamera:
    """Unified (MEI) omnidirectional camera model used by KITTI-360's fisheyes."""
    xi: float
    k1: float
    k2: float
    p1: float
    p2: float
    gamma1: float
    gamma2: float
    u0: float
    v0: float
    width: int
    height: int

    @classmethod
    def from_yaml(cls, path: str) -> "MEICamera":
        vals: dict[str, float] = {}
        with open(path) as f:
            for line in f:
                m = re.match(r"\s*(\w+)\s*:\s*([-+0-9.eE]+)\s*$", line)
                if m:
                    vals[m.group(1)] = float(m.group(2))
        need = ["xi", "k1", "k2", "p1", "p2", "gamma1", "gamma2", "u0", "v0",
                "image_width", "image_height"]
        missing = [k for k in need if k not in vals]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        return cls(vals["xi"], vals["k1"], vals["k2"], vals["p1"], vals["p2"],
                   vals["gamma1"], vals["gamma2"], vals["u0"], vals["v0"],
                   int(vals["image_width"]), int(vals["image_height"]))

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """[N,3] rays in the fisheye camera frame → ([N] u, [N] v); NaN if not visible."""
        n = np.linalg.norm(rays, axis=1, keepdims=True)
        x, y, z = (rays / np.maximum(n, 1e-12)).T
        denom = z + self.xi
        ok = denom > 1e-6
        mx = np.where(ok, x / np.where(ok, denom, 1), np.nan)
        my = np.where(ok, y / np.where(ok, denom, 1), np.nan)
        r2 = mx * mx + my * my
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2
        dx = 2 * self.p1 * mx * my + self.p2 * (r2 + 2 * mx * mx)
        dy = self.p1 * (r2 + 2 * my * my) + 2 * self.p2 * mx * my
        u = self.gamma1 * (mx * radial + dx) + self.u0
        v = self.gamma2 * (my * radial + dy) + self.v0
        inside = ok & (u >= 0) & (u < self.width - 1) & (v >= 0) & (v < self.height - 1)
        return np.where(inside, u, np.nan), np.where(inside, v, np.nan)


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


class Kitti360ViewRenderer:
    """Render the three Qwen-Drive views for any frame of a KITTI-360 drive.

    side_yaw_deg / side_hfov_deg: direction and width of the virtual side
    cameras (degrees from straight ahead; KITTI-360's front camera covers
    roughly ±52°). side_size: (width, height) of the rendered side views.
    """

    def __init__(
        self,
        root: str,
        sequence: str,
        side_mode: str = "fisheye",
        side_yaw_deg: float = 55.0,
        side_hfov_deg: float = 75.0,
        side_size: tuple[int, int] = (960, 640),
    ) -> None:
        if side_mode not in ("fisheye", "front_crop", "blank"):
            raise ValueError(f"unknown side_mode {side_mode!r}")
        self.root = root
        self.sequence = sequence
        self.side_mode = side_mode
        self.side_yaw = side_yaw_deg
        self.side_hfov = side_hfov_deg
        self.side_size = side_size
        self._maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if side_mode == "fisheye":
            self._init_fisheye_maps()

    # ── paths ─────────────────────────────────────────────────────────────────

    def front_path(self, frame_idx: int) -> str:
        return os.path.join(self.root, "data_2d_raw", self.sequence, "image_00",
                            "data_rect", f"{frame_idx:010d}.png")

    def fisheye_path(self, cam: str, frame_idx: int) -> str:
        return os.path.join(self.root, "data_2d_raw", self.sequence, cam,
                            "data_rgb", f"{frame_idx:010d}.png")

    # ── fisheye remap ─────────────────────────────────────────────────────────

    def _init_fisheye_maps(self) -> None:
        calib = os.path.join(self.root, "calibration")
        from fusion_perception.data.kitti_calibration import _parse_cam_to_pose_360

        c2p = os.path.join(calib, "calib_cam_to_pose.txt")
        R_pose_cam0 = _parse_cam_to_pose_360(c2p, "image_00")[:3, :3].astype(np.float64)
        w, h = self.side_size
        f = (w / 2) / np.tan(np.deg2rad(self.side_hfov) / 2)
        us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays_v = np.stack([(us - w / 2) / f, (vs - h / 2) / f, np.ones_like(us)], -1).reshape(-1, 3)
        # yaw > 0 turns left; in camera axes (x right, y down) that is a
        # rotation about y by -yaw
        for tag, cam, yaw in ((FRONT_LEFT, "image_02", self.side_yaw),
                              (FRONT_RIGHT, "image_03", -self.side_yaw)):
            model = MEICamera.from_yaml(os.path.join(calib, f"{cam}.yaml"))
            R_pose_fish = _parse_cam_to_pose_360(c2p, cam)[:3, :3].astype(np.float64)
            R = R_pose_fish.T @ R_pose_cam0 @ _rot_y(-yaw)
            u, v = model.project(rays_v @ R.T)
            self._maps[tag] = (u.reshape(h, w).astype(np.float32),
                               v.reshape(h, w).astype(np.float32))
            frac = float(np.isfinite(u).mean())
            if frac < 0.9:
                logger.warning(f"{tag}: only {100 * frac:.0f}% of the virtual view "
                               f"is covered by {cam}; reduce side_yaw/side_hfov")

    # ── rendering ─────────────────────────────────────────────────────────────

    @staticmethod
    def _read_rgb(path: str) -> Optional[np.ndarray]:
        bgr = cv2.imread(path)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def render(self, frame_idx: int) -> Optional[dict[str, np.ndarray]]:
        """{view tag: RGB uint8 image} for one timestamp, or None if the front image is missing."""
        front = self._read_rgb(self.front_path(frame_idx))
        if front is None:
            return None
        views = {FRONT: front}
        if self.side_mode == "fisheye":
            for tag, cam in ((FRONT_LEFT, "image_02"), (FRONT_RIGHT, "image_03")):
                img = self._read_rgb(self.fisheye_path(cam, frame_idx))
                if img is None:
                    return None
                mx, my = self._maps[tag]
                views[tag] = cv2.remap(img, np.nan_to_num(mx, nan=-1), np.nan_to_num(my, nan=-1),
                                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))
        elif self.side_mode == "front_crop":
            w = front.shape[1]
            views[FRONT_LEFT] = front[:, : w // 2].copy()
            views[FRONT_RIGHT] = front[:, w // 2:].copy()
        else:
            grey = np.full((self.side_size[1], self.side_size[0], 3), 128, np.uint8)
            views[FRONT_LEFT] = grey
            views[FRONT_RIGHT] = grey.copy()
        return views

    def render_history(self, frame_idx: int, offsets: tuple[int, ...]) -> Optional[dict[str, list[np.ndarray]]]:
        """{view tag: [image per offset, oldest first]}; None if any frame is missing."""
        out: dict[str, list[np.ndarray]] = {t: [] for t in VIEW_TAGS}
        for k in offsets:
            views = self.render(frame_idx + k)
            if views is None:
                return None
            for t in VIEW_TAGS:
                out[t].append(views[t])
        return out
