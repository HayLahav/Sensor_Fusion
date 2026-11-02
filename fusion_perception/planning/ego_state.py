"""Ego state for a planner, computed from KITTI-360 poses.

For each frame this gives, in the current ego frame (x forward, y left):
  - speed and acceleration (vx, vy, ax, ay) from finite differences
  - past path: 16 points over 1.5 s at 10 Hz, plus the current pose
  - future path actually driven: 50 points over 5 s (ground truth, used for
    evaluation and for the turn command)
  - turn command (left / straight / right / unknown), derived from the
    heading change along the future path, the way benchmarks like NAVSIM
    derive it from the route

The camera frame (x right, y down, z forward) has known axes, so poses are
taken as camera→world and converted: forward = z_cam, left = -x_cam.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from fusion_perception.tracking.ego_motion import PoseEgoCompensation

LEFT, STRAIGHT, RIGHT, UNKNOWN = "left", "straight", "right", "unknown"
_COMMAND_ORDER = (LEFT, STRAIGHT, RIGHT, UNKNOWN)


@dataclass
class EgoState:
    frame_idx: int
    velocity: list[float]                 # [vx, vy] m/s, ego frame
    acceleration: list[float]             # [ax, ay] m/s², ego frame
    speed: float
    history: list[list[float]]            # [[x, y, heading], ...] oldest → current (last is [0,0,0])
    future: Optional[list[list[float]]]   # [[x, y, heading], ...] t = dt … horizon, None if unavailable
    command: str                          # left | straight | right | unknown
    command_heading_change: Optional[float] = None   # rad, what the command was derived from
    extra: dict = field(default_factory=dict)

    def command_one_hot(self) -> list[float]:
        """[left, straight, right, unknown] one-hot."""
        return [1.0 if c == self.command else 0.0 for c in _COMMAND_ORDER]

    def to_qwen_drive(self) -> dict:
        """Ego fields in the layout described in Qwen-Drive's docs/data.md.

        Check the field names against the Qwen-Drive version you install
        before wiring this into its scene files.
        """
        nav = {STRAIGHT: 0, LEFT: 1, RIGHT: 2}.get(self.command, 0)
        return {
            "ego_status": {
                "ego_velocity": list(self.velocity),
                "ego_acceleration": list(self.acceleration),
                "driving_command": self.command_one_hot(),
            },
            "nav_command": nav,
            "history_trajectory": [list(p) for p in self.history],
            # Planning expert input: [vx, vy, ax, ay, *driving_command]
            "expert_ego_vector": list(self.velocity) + list(self.acceleration)
            + self.command_one_hot(),
        }


def _cam_to_ego_xy(p_cam: np.ndarray) -> np.ndarray:
    """[..., 3] camera-frame points → [..., 2] ego (forward, left)."""
    return np.stack([p_cam[..., 2], -p_cam[..., 0]], axis=-1)


def _heading_in(T_cur_from: np.ndarray) -> float:
    """Heading of another camera pose, expressed in the current ego frame."""
    fwd = T_cur_from[:3, 2]                     # that camera's z-axis in current cam
    return float(np.arctan2(-fwd[0], fwd[2]))   # left = -x_cam, forward = z_cam


class EgoStateExtractor:
    """Compute EgoState for any frame of a KITTI-360 drive.

    Parameters
    ----------
    ego_comp : PoseEgoCompensation (gives camera→world for any frame,
               interpolating frames missing from poses.txt)
    fps      : frame rate of the sequence (KITTI-360: 10 Hz)
    history_s, future_s : lengths of the past and future paths (seconds)
    turn_threshold_deg  : heading change along the future path that counts
                          as a turn
    turn_lookahead_m    : look at most this far along the future path when
                          deciding the command (so a turn 5 s away at high
                          speed doesn't trigger it too early)
    """

    def __init__(
        self,
        ego_comp: PoseEgoCompensation,
        fps: float = 10.0,
        history_s: float = 1.5,
        future_s: float = 5.0,
        turn_threshold_deg: float = 20.0,
        turn_lookahead_m: float = 30.0,
        min_speed_for_turn: float = 0.5,
    ) -> None:
        self.ego_comp = ego_comp
        self.fps = fps
        self.n_hist = int(round(history_s * fps))
        self.n_future = int(round(future_s * fps))
        self.turn_threshold = np.deg2rad(turn_threshold_deg)
        self.turn_lookahead_m = turn_lookahead_m
        self.min_speed_for_turn = min_speed_for_turn

    # ── helpers ───────────────────────────────────────────────────────────────

    def _relative(self, frame_idx: int, other: int) -> Optional[np.ndarray]:
        """4×4 transform: camera at `other` → camera at `frame_idx`."""
        T_cur = self.ego_comp.rectcam_to_world(frame_idx)
        T_oth = self.ego_comp.rectcam_to_world(other)
        if T_cur is None or T_oth is None:
            return None
        return np.linalg.inv(T_cur) @ T_oth

    def _path(self, frame_idx: int, offsets: range) -> Optional[np.ndarray]:
        pts = []
        for k in offsets:
            T = self._relative(frame_idx, frame_idx + k)
            if T is None:
                return None
            xy = _cam_to_ego_xy(T[:3, 3])
            pts.append([float(xy[0]), float(xy[1]), _heading_in(T)])
        return np.array(pts, dtype=np.float64)

    def _derive_command(self, future: Optional[np.ndarray], speed: float) -> tuple[str, Optional[float]]:
        if future is None or len(future) == 0:
            return UNKNOWN, None
        if speed < self.min_speed_for_turn and np.linalg.norm(future[-1, :2]) < 2.0:
            return STRAIGHT, 0.0                      # standing still
        seg = np.linalg.norm(np.diff(np.vstack([[0, 0], future[:, :2]]), axis=0), axis=1)
        dist = np.cumsum(seg)
        idx = int(np.searchsorted(dist, self.turn_lookahead_m))
        idx = min(idx, len(future) - 1)
        dh = float(future[idx, 2])
        if dh > self.turn_threshold:
            return LEFT, dh
        if dh < -self.turn_threshold:
            return RIGHT, dh
        return STRAIGHT, dh

    # ── main entry ────────────────────────────────────────────────────────────

    def get(self, frame_idx: int) -> Optional[EgoState]:
        """EgoState for frame_idx, or None if poses don't cover the needed past."""
        dt = 1.0 / self.fps
        hist = self._path(frame_idx, range(-self.n_hist, 1))
        if hist is None:
            return None
        future = self._path(frame_idx, range(1, self.n_future + 1))

        # Velocity: central difference when the next frame is known, else backward
        p_prev = hist[-2, :2]
        p_prev2 = hist[-3, :2]
        if future is not None:
            v = (future[0, :2] - p_prev) / (2 * dt)
            a = (future[0, :2] - 2 * hist[-1, :2] + p_prev) / dt ** 2
        else:
            v = (hist[-1, :2] - p_prev) / dt
            a = (hist[-1, :2] - 2 * p_prev + p_prev2) / dt ** 2
        speed = float(np.linalg.norm(v))
        command, dh = self._derive_command(future, speed)

        return EgoState(
            frame_idx=frame_idx,
            velocity=[float(v[0]), float(v[1])],
            acceleration=[float(a[0]), float(a[1])],
            speed=speed,
            history=hist.tolist(),
            future=future.tolist() if future is not None else None,
            command=command,
            command_heading_change=dh,
        )
