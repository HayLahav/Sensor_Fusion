"""Per-frame ego state + BEV check of the path the car actually drove.

The driven path is safe by definition, so any BEV-check violation on it is a
false alarm. Running this over a KITTI-360 drive measures how much to trust
the check before it is used on a planner's paths (e.g. Qwen-Drive).

Writes planning.json: per-frame ego state (Qwen-Drive layout), command, and
check result, plus a summary with the false-alarm rate per check type.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Optional
import numpy as np

from fusion_perception.planning.bev_check import (
    BEVCheckConfig, build_plan_context, check_trajectory,
)
from fusion_perception.planning.ego_state import EgoStateExtractor
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("planning_audit")


class DrivenPathAudit:
    def __init__(
        self,
        extractor: EgoStateExtractor,
        cfg: Optional[BEVCheckConfig] = None,
        every: int = 1,
    ) -> None:
        self.extractor = extractor
        self.cfg = cfg or BEVCheckConfig()
        self.every = max(1, every)
        self.records: list[dict] = []

    def update(self, frame_idx: int, outputs: dict,
               lidar_pts_cam_all: Optional[np.ndarray]) -> Optional[dict]:
        if frame_idx % self.every:
            return None
        try:
            state = self.extractor.get(frame_idx)
            if state is None:
                return None
            rec = {
                "frame_idx": frame_idx,
                "speed": state.speed,
                "command": state.command,
                "command_heading_change": state.command_heading_change,
                "ego": state.to_qwen_drive(),
                "driven_path": state.future,
                "driven_path_check": None,
            }
            if state.future is not None:
                ctx = build_plan_context(
                    outputs.get("tracks", []),
                    occupancy=outputs.get("occupancy"),
                    lidar_pts_cam=lidar_pts_cam_all,
                    lidar_labels=outputs.get("lidar_labels"),
                    cfg=self.cfg,
                )
                res = check_trajectory(np.asarray(state.future), ctx,
                                       dt=1.0 / self.extractor.fps, cfg=self.cfg)
                rec["driven_path_check"] = res.to_dict()
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
        return {
            "frames_with_ego_state": len(self.records),
            "frames_checked": len(checked),
            "false_alarm_rate": len(flagged) / n,
            "false_alarm_rate_by_kind": {k: c / n for k, c in sorted(kinds.items())},
            "commands": dict(Counter(r["command"] for r in self.records)),
            "flagged_frames": [r["frame_idx"] for r in flagged][:200],
        }

    def save(self, path: Path) -> dict:
        summ = self.summary()
        Path(path).write_text(json.dumps({"summary": summ, "frames": self.records}, indent=1))
        return summ
