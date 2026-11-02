"""Per-frame ego state + BEV check of the driven path and of a planner's paths.

Driven path: the path the car actually drove is safe by definition, so any
BEV-check violation on it is a false alarm. The false-alarm rate says how far
to trust the check.

Planner path (optional): for frames where a plan exists (Qwen-Drive
predictions file, or the in-process planner), the same check runs on every
sampled path, plus ADE / FDE against the driven path. A planner flag is only
meaningful relative to the driven-path false-alarm rate.

Writes planning.json: per-frame records and a summary.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Optional
import numpy as np

from fusion_perception.planning.bev_check import (
    BEVCheckConfig, build_plan_context, check_trajectory,
)
from fusion_perception.planning.ego_state import EgoState, EgoStateExtractor
from fusion_perception.planning.metrics import displacement_errors
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("planning_audit")

# (frame_idx, ego state) → (trajectories [S,N,3], reasoning) or None
PlanSource = Callable[[int, EgoState], Optional[tuple[np.ndarray, Optional[str]]]]


class PredictionsFileSource:
    """Plans from Qwen-Drive's run_planning.py output (matched by scene token)."""

    def __init__(self, path: str, sequence: str) -> None:
        from fusion_perception.planning.qwen_drive_scene import read_predictions, scene_token

        self._preds = read_predictions(path)
        self._token = lambda f: scene_token(sequence, f)
        logger.info(f"Loaded {len(self._preds)} Qwen-Drive predictions from {path}")

    def __call__(self, frame_idx: int, state: EgoState):
        rec = self._preds.get(self._token(frame_idx))
        if rec is None or not rec.get("trajectories"):
            return None
        return np.asarray(rec["trajectories"], dtype=np.float64), rec.get("reasoning")


class LivePlannerSource:
    """Runs the in-process Qwen-Drive planner every `every` frames."""

    def __init__(self, planner, renderer, fps: float, every: int = 10) -> None:
        from fusion_perception.planning.qwen_drive_scene import camera_frame_offsets

        self.planner = planner
        self.renderer = renderer
        self.offsets = camera_frame_offsets(fps)
        self.every = max(1, every)

    def __call__(self, frame_idx: int, state: EgoState):
        if frame_idx % self.every:
            return None
        images = self.renderer.render_history(frame_idx, self.offsets)
        if images is None:
            return None
        out = self.planner.plan(state, images, token=str(frame_idx))
        return out.trajectories, out.reasoning


class PlanningAudit:
    def __init__(
        self,
        extractor: EgoStateExtractor,
        cfg: Optional[BEVCheckConfig] = None,
        every: int = 1,
        plan_source: Optional[PlanSource] = None,
        default_grid: Optional[dict] = None,
    ) -> None:
        """default_grid: {resolution, x_range, z_range} used when a frame's
        outputs carry no OccupancyGrid."""
        self.default_grid = default_grid or {"resolution": 0.5, "x_range": [-25.0, 25.0],
                                             "z_range": [-10.0, 60.0]}
        self.extractor = extractor
        self.cfg = cfg or BEVCheckConfig()
        self.every = max(1, every)
        self.plan_source = plan_source
        self.records: list[dict] = []

    def update(self, frame_idx: int, outputs: dict,
               lidar_pts_cam_all: Optional[np.ndarray]) -> Optional[dict]:
        plan = None
        try:
            state = self.extractor.get(frame_idx)
            if state is None:
                return None
            if self.plan_source is not None:
                plan = self.plan_source(frame_idx, state)
            if frame_idx % self.every and plan is None:
                return None
            rec = {
                "frame_idx": frame_idx,
                "speed": state.speed,
                "command": state.command,
                "command_heading_change": state.command_heading_change,
                "ego": state.to_qwen_drive(),
                "driven_path": state.future,
                "driven_path_check": None,
                "planner": None,
            }
            driven = state.valid_future()
            has_driven = driven is not None and len(driven) > 0
            if not has_driven and plan is None:
                self.records.append(rec)
                return rec

            ctx = build_plan_context(
                outputs.get("tracks", []),
                occupancy=outputs.get("occupancy"),
                lidar_pts_cam=lidar_pts_cam_all,
                lidar_labels=outputs.get("lidar_labels"),
                cfg=self.cfg,
                **self.default_grid,
            )
            dt = 1.0 / self.extractor.fps
            if has_driven:
                rec["driven_path_check"] = check_trajectory(driven, ctx, dt=dt, cfg=self.cfg).to_dict()
            if plan is not None:
                trajs, reasoning = plan
                samples = []
                for traj in np.asarray(trajs):
                    s = {"check": check_trajectory(traj, ctx, dt=dt, cfg=self.cfg).to_dict()}
                    if has_driven:
                        s["metrics"] = displacement_errors(traj, driven, dt=dt)
                    samples.append(s)
                rec["planner"] = {
                    "reasoning": reasoning,
                    "trajectories": np.round(np.asarray(trajs), 4).tolist(),
                    "samples": samples,
                }
            self.records.append(rec)
            return rec
        except Exception as exc:          # never break the perception loop
            logger.warning(f"planning audit failed at frame {frame_idx}: {exc}")
            return None

    def summary(self) -> dict:
        checked = [r for r in self.records if r["driven_path_check"] is not None]
        flagged = [r for r in checked if not r["driven_path_check"]["ok"]]
        kinds: Counter = Counter()
        for r in flagged:
            kinds.update({v["kind"] for v in r["driven_path_check"]["violations"]})
        n = max(len(checked), 1)
        out = {
            "frames_with_ego_state": len(self.records),
            "frames_checked": len(checked),
            "false_alarm_rate": len(flagged) / n,
            "false_alarm_rate_by_kind": {k: c / n for k, c in sorted(kinds.items())},
            "commands": dict(Counter(r["command"] for r in self.records)),
            "flagged_frames": [r["frame_idx"] for r in flagged][:200],
        }

        planned = [r for r in self.records if r["planner"] and r["planner"]["samples"]]
        if planned:
            first = [r["planner"]["samples"][0] for r in planned]
            p_flag = [s for s in first if not s["check"]["ok"]]
            p_kinds: Counter = Counter()
            for s in p_flag:
                p_kinds.update({v["kind"] for v in s["check"]["violations"]})
            m = len(planned)
            metric_keys = sorted({k for s in first for k in s.get("metrics", {})})
            mean = {k: float(np.mean([s["metrics"][k] for s in first if k in s.get("metrics", {})]))
                    for k in metric_keys}
            # Best of the sampled paths, by ADE@3s (or the longest horizon available)
            best_key = next((k for k in ("ADE@3s", "ADE@2s", "ADE@1s") if k in metric_keys), None)
            best = None
            if best_key:
                vals = [min(s["metrics"][best_key] for s in r["planner"]["samples"]
                            if best_key in s.get("metrics", {}))
                        for r in planned
                        if any(best_key in s.get("metrics", {}) for s in r["planner"]["samples"])]
                best = {f"min_{best_key}": float(np.mean(vals))} if vals else None
            out["planner"] = {
                "frames": m,
                "flag_rate": len(p_flag) / m,
                "flag_rate_by_kind": {k: c / m for k, c in sorted(p_kinds.items())},
                "metrics_first_sample": mean,
                "metrics_best_sample": best,
                "flagged_frames": [r["frame_idx"] for r, s in zip(planned, first)
                                   if not s["check"]["ok"]][:200],
            }
        return out

    def save(self, path: Path) -> dict:
        summ = self.summary()
        Path(path).write_text(json.dumps({"summary": summ, "frames": self.records}, indent=1))
        return summ


# Backwards-compatible name
DrivenPathAudit = PlanningAudit
