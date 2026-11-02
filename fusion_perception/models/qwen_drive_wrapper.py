"""Qwen-Drive-1.0 planner wrapper (in-process).

Qwen-Drive (https://github.com/QwenLM/Qwen-Drive-1.0, Apache-2.0) plans a 5 s
ego path (50 points of x, y, heading at 10 Hz) from three forward cameras and
the ego state.

Environment: Qwen-Drive pins torch 2.8 / transformers 5.x; this project pins
torch 2.5.1 for WildDet3D, CoTracker and Gemma. Use this wrapper only if both
work in one environment. Otherwise use the two-step route:
  1. scripts/export_qwen_drive_scenes.py   (this env)   → scenes.jsonl + frames
  2. Qwen-Drive's scripts/run_planning.py  (its env)    → predictions.jsonl
  3. inference.py with planning.qwen_drive.predictions  → BEV check + metrics

Weights: Qwen/Qwen-Drive-1.0-4B (VLM ≈ 9 GB) plus planner-rl or planner-sft
(≈ 2 GB). planner-rl is meant for REASONING_PLANNING; planner-sft works in
both modes. A 24 GB GPU is recommended; on a 16 GB T4 use float16 (no
bfloat16) and attention "sdpa" (no FlashAttention), and expect it to be tight.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from fusion_perception.planning.ego_state import EgoState
from fusion_perception.planning.qwen_drive_scene import driving_scene
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("qwen_drive_wrapper")


@dataclass
class PlannerOutput:
    trajectories: np.ndarray        # [num_samples, 50, 3] ego frame (x fwd, y left, heading)
    reasoning: Optional[str]

    @property
    def trajectory(self) -> np.ndarray:
        return self.trajectories[0]


class QwenDrivePlanner:
    def __init__(
        self,
        model_path: str = "Qwen-Drive-1.0-4B",
        planner_path: Optional[str] = "Qwen-Drive-1.0-4B/planner-rl",
        mode: str = "reasoning_planning",
        num_samples: int = 1,
        dtype: str = "auto",                 # auto | bfloat16 | float16 | float32
        attn_implementation: str = "sdpa",   # flash_attention_2 only on Ampere+
        device: str = "cuda",
        seed: Optional[int] = None,
    ) -> None:
        self.model_path = model_path
        self.planner_path = planner_path
        self.mode = mode
        self.num_samples = num_samples
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.device = device
        self.seed = seed
        self._model = None

    def _torch_dtype(self):
        import torch

        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        if self.device.startswith("cuda") and torch.cuda.is_available() \
                and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16 if self.device.startswith("cuda") else torch.float32

    def load(self) -> None:
        from qwen_drive import QwenDriveForPlanning

        dtype = self._torch_dtype()
        model = QwenDriveForPlanning.from_pretrained(
            self.model_path, planner=self.planner_path, dtype=dtype,
            attn_implementation=self.attn_implementation,
        )
        self._model = model.to(self.device).eval()
        logger.info(f"Qwen-Drive loaded ({self.model_path}, planner={self.planner_path}, "
                    f"{dtype}, {self.attn_implementation}, mode={self.mode})")

    def plan(self, state: EgoState, images: dict[str, list[Any]], token: str = "") -> PlannerOutput:
        """images: {view tag: [4 frames, oldest first]} (PIL images or RGB arrays)."""
        if self._model is None:
            raise RuntimeError("QwenDrivePlanner.load() must be called first")
        scene = driving_scene(state, images, token=token)
        result = self._model.run(self.mode, scene=scene, num_samples=self.num_samples,
                                 seed=self.seed)
        return PlannerOutput(np.asarray(result.trajectories, dtype=np.float64), result.reasoning)

    def unload(self) -> None:
        self._model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
