"""Ego state for a planner, computed from KITTI-360 poses.

For each frame this gives, in the current ego frame (x forward, y left):
  - speed and acceleration (vx, vy, ax, ay) from finite differences
  - past path: 16 points over 1.5 s at 10 Hz, the last one being the current
    pose (0, 0, 0), with velocity and acceleration at each point expressed
    in that point's own body frame (Qwen-Drive's convention, docs/data.md)
  - future path actually driven: 50 points over 5 s (ground truth, used for
    evaluation and for the turn command), with a validity mask near the end
    of a drive
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
    history_velocity: list[list[float]] = field(default_factory=list)      # [16, 2] body frame
    history_acceleration: list[list[float]] = field(default_factory=list)  # [16, 2] body frame
    future_valid: Optional[list[int]] = None   # [50] 1 where the future pose is real
    extra: dict = field(default_factory=dict)

    @property
    def nav_command(self) -> int:
        """Qwen-Drive scalar command: 0 straight, 1 left, 2 right (unknown → straight)."""
        return {STRAIGHT: 0, LEFT: 1, RIGHT: 2}.get(self.command, 0)

    def valid_future(self) -> Optional[np.ndarray]:
        """Future path cut at the first invalid point (end of the drive)."""
        if self.future is None:
            return None
        fut = np.asarray(self.future)
        if self.future_valid is None:
            return fut
        bad = np.flatnonzero(np.asarray(self.future_valid) == 0)
        return fut[: bad[0]] if len(bad) else fut

    def command_one_hot(self) -> list[float]:
        """[left, straight, right, unknown] one-hot."""
        return [1.0 if c == self.command else 0.0 for c in _COMMAND_ORDER]

    def to_qwen_drive(self) -> dict:
        """Ego fields in the layout described in Qwen-Drive's docs/data.md.

        Check the field names against the Qwen-Drive version you install
        before wiring this into its scene files.
        """
        return {
            "ego_status": {
                "ego_velocity": list(self.velocity),
                "ego_acceleration": list(self.acceleration),
                "driving_command": self.command_one_hot(),
            },
            "nav_command": self.nav_command,
            "hist_traj_10hz": [list(p) for p in self.history],
            "hist_vel_10hz": [list(v) for v in self.history_velocity],
            "hist_acc_10hz": [list(a) for a in self.history_acceleration],
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

    def _pose(self, frame_idx: int, k: int) -> Optional[list[float]]:
        T = self._relative(frame_idx, frame_idx + k)
        if T is None:
            return None
        xy = _cam_to_ego_xy(T[:3, 3])
        return [float(xy[0]), float(xy[1]), _heading_in(T)]

    def _path(self, frame_idx: int, offsets: range) -> Optional[np.ndarray]:
        pts = [self._pose(frame_idx, k) for k in offsets]
        if any(p is None for p in pts):
            return None
        return np.array(pts, dtype=np.float64)

    def _future(self, frame_idx: int) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        pts = [self._pose(frame_idx, k) for k in range(1, self.n_future + 1)]
        valid = np.array([p is not None for p in pts], dtype=np.int64)
        if not valid.any():
            return None, None
        last = None
        filled = []
        for p in pts:                 # pad the end of a drive with the last real pose
            last = p if p is not None else last
            filled.append(last if last is not None else [0.0, 0.0, 0.0])
        return np.array(filled, dtype=np.float64), valid

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
        # 16 poses over 1.5 s: offsets -15 … 0
        hist = self._path(frame_idx, range(-self.n_hist, 1))
        if hist is None:
            return None
        future, future_valid = self._future(frame_idx)

        # Velocity / acceleration by finite differences over the history, with
        # up to two extra poses on each side (acceleration differences the
        # velocity again, so central differences need two).
        before = []
        for k in (-self.n_hist - 1, -self.n_hist - 2):
            p = self._pose(frame_idx, k)
            if p is None:
                break
            before.insert(0, p)
        after = []
        if future is not None:
            for i in range(min(2, len(future))):
                if not future_valid[i]:
                    break
                after.append(future[i].tolist())
        series = np.asarray(before + hist.tolist() + after)
        start = len(before)
        vel = np.gradient(series[:, :2], dt, axis=0)
        acc = np.gradient(vel, dt, axis=0)
        vel, acc = vel[start:start + len(hist)], acc[start:start + len(hist)]
        # Express each in the body frame of its own timestamp
        c, s_ = np.cos(hist[:, 2]), np.sin(hist[:, 2])
        rot = lambda q: np.stack([c * q[:, 0] + s_ * q[:, 1], -s_ * q[:, 0] + c * q[:, 1]], 1)
        vel_b, acc_b = rot(vel), rot(acc)
        v, a = vel_b[-1], acc_b[-1]           # current heading is 0
        speed = float(np.linalg.norm(v))
        valid_future = None if future is None else future[: int(future_valid.cumprod().sum())]
        command, dh = self._derive_command(valid_future, speed)

        return EgoState(
            frame_idx=frame_idx,
            velocity=[float(v[0]), float(v[1])],
            acceleration=[float(a[0]), float(a[1])],
            speed=speed,
            history=hist.tolist(),
            future=future.tolist() if future is not None else None,
            command=command,
            command_heading_change=dh,
            history_velocity=vel_b.tolist(),
            history_acceleration=acc_b.tolist(),
            future_valid=future_valid.tolist() if future_valid is not None else None,
        )
