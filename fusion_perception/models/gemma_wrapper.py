"""Gemma 4 2B reasoning wrapper.

Visual mode: sends a rolling window of annotated frames alongside the text
prompt, with a configurable token budget per image (default 70 tokens ≈ one
low-resolution crop — no Pan-and-Scan tiling).

Text mode: prompt only — faster, lower memory.

Token budget guide (visual_token_budget param):
  70   → ~117×117 px resize + max_num_tiles=1  (ultra-low latency)
  140  → ~166×166 px resize + max_num_tiles=1
  280  → ~235×235 px resize + max_num_tiles=1
  560  → ~332×332 px resize + max_num_tiles=1
  1120 → full 224×224 + Pan-and-Scan (max quality)
"""
from __future__ import annotations
import math
import time
from collections import deque
import numpy as np
from PIL import Image
from typing import Optional
from fusion_perception.utils.dataclasses import SceneMemory, ReasoningOutput
from fusion_perception.reasoning.prompt_builder import build_scene_prompt
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("gemma_wrapper")

# Gemma 4 uses SigLIP with 14×14-pixel patches.
# n_tokens ≈ (img_px / 14)² → img_px ≈ 14 × √n_tokens
_PATCH_PX = 14


def _budget_to_px(token_budget: int) -> int:
    """Target image edge size (pixels) for a given token budget."""
    return max(28, int(_PATCH_PX * math.sqrt(token_budget)))


class GemmaReasoningWrapper:
    """Wraps Gemma 4 2B-IT for semantic scene reasoning.

    Visual mode buffers annotated frames from every reasoning call and sends
    the last `rolling_frames` resized images alongside the text prompt.
    Token cost per image is controlled via `visual_token_budget`.

    Call load() once, then reason() per trigger.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-2b-it",
        quantize_4bit: bool = True,
        max_new_tokens: int = 100,
        device: str = "cuda",
        visual_mode: bool = False,
        rolling_frames: int = 3,
        visual_token_budget: int = 70,
    ) -> None:
        self.model_id = model_id
        self.quantize_4bit = quantize_4bit
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.visual_mode = visual_mode
        self.rolling_frames = rolling_frames
        self.visual_token_budget = visual_token_budget
        self._img_px = _budget_to_px(visual_token_budget)
        self._model = None
        self._processor = None
        # Rolling buffer of resized PIL frames — filled regardless of visual_mode
        # so the buffer is warm if visual_mode is toggled at runtime.
        self._frame_buffer: deque[Image.Image] = deque(maxlen=rolling_frames)

    def load(self) -> None:
        """Load Gemma model with optional INT4 quantization."""
        logger.info(
            f"Loading {self.model_id} | 4bit={self.quantize_4bit} | "
            f"visual={self.visual_mode} | "
            f"{self.rolling_frames} frames × {self.visual_token_budget} tok "
            f"({self._img_px}×{self._img_px} px)"
        )
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        kwargs: dict = {"device_map": self.device}
        if self.quantize_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        log_gpu_memory("Gemma loaded")
        logger.info("Gemma ready")

    def reason(
        self,
        memory: SceneMemory,
        annotated_frame: Optional[np.ndarray],
        trigger_reason: str = "interval",
        fps: float = 30.0,
        memory_prefix: str = "",
    ) -> Optional[ReasoningOutput]:
        """Run Gemma on the current scene state.

        annotated_frame : RGB [H,W,3] uint8 annotated camera frame, or None.
                          Always added to the rolling buffer when provided;
                          only used visually when visual_mode=True.
        memory_prefix   : compact past-observation text from GemmaMemoryBuffer.
        """
        if self._model is None:
            return None

        # ── Buffer frame (always, visual_mode independent) ────────────────────
        if annotated_frame is not None:
            pil = Image.fromarray(annotated_frame).resize(
                (self._img_px, self._img_px), Image.LANCZOS
            )
            self._frame_buffer.append(pil)

        use_visual = self.visual_mode and len(self._frame_buffer) > 0
        prompt_text = build_scene_prompt(memory, fps=fps, memory_prefix=memory_prefix)

        t0 = time.perf_counter()
        if use_visual:
            raw = self._run_visual(prompt_text, list(self._frame_buffer))
        else:
            raw = self._run_text(prompt_text)
        latency_ms = (time.perf_counter() - t0) * 1000

        summary, anomalies, trajectory_nl = self._parse_response(raw)

        if use_visual:
            mode_str = (f"visual ×{len(self._frame_buffer)} "
                        f"@{self.visual_token_budget}tok/{self._img_px}px")
        else:
            mode_str = "text"
        logger.info(
            f"Frame {memory.frame_idx} reasoning: {latency_ms:.0f}ms [{mode_str}]"
        )

        return ReasoningOutput(
            frame_idx=memory.frame_idx,
            trigger_reason=trigger_reason,
            visual_mode=use_visual,
            prompt_used=prompt_text,
            summary=summary,
            anomalies=anomalies,
            trajectory_nl=trajectory_nl,
            raw_response=raw,
            latency_ms=latency_ms,
        )

    # ── Inference back-ends ───────────────────────────────────────────────────

    def _run_text(self, prompt: str) -> str:
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        return self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _run_visual(self, prompt: str, frames: list[Image.Image]) -> str:
        """Multi-frame visual inference with token budget control.

        Sends `frames` (oldest → newest) as image tokens followed by the text
        prompt.  `max_num_tiles=1` disables Pan-and-Scan tiling so each image
        contributes ~visual_token_budget tokens instead of the full 1120.
        """
        import torch

        # Build content list: [img_t-N, ..., img_t, text]
        content = [{"type": "image", "image": f} for f in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        proc_kwargs = dict(text=text, images=frames, return_tensors="pt")
        try:
            # max_num_tiles=1 → base crop only, no extra tiling slices
            inputs = self._processor(**proc_kwargs, max_num_tiles=1).to(self.device)
        except TypeError:
            # Processor version does not support max_num_tiles — resize alone limits tokens
            inputs = self._processor(**proc_kwargs).to(self.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        return self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> tuple[str, list[str], str]:
        """Extract SUMMARY / ANOMALY / TRAJECTORY from structured output.

        Falls back to first-line heuristic if the model ignores the format.
        """
        summary = trajectory_nl = ""
        anomalies: list[str] = []

        for line in raw.strip().splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("SUMMARY:"):
                summary = stripped[8:].strip()
            elif upper.startswith("CAUTION:"):
                text = stripped[8:].strip()
                if text and text.lower() not in ("none", "none.", "n/a"):
                    anomalies = [text]
            elif upper.startswith("ANOMALY:"):   # backward compat
                text = stripped[8:].strip()
                if text and text.lower() not in ("none", "none.", "n/a"):
                    anomalies = [text]
            elif upper.startswith("TRAJECTORY:"):
                trajectory_nl = stripped[11:].strip()

        if not summary:
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            summary = lines[0] if lines else raw
            anomalies = [l.lstrip("•-* ") for l in lines
                         if l.startswith(("•", "-", "*"))]
            trajectory_nl = " ".join(lines[1:]) if len(lines) > 1 else ""

        return summary, anomalies, trajectory_nl

    def unload(self) -> None:
        import torch
        self._frame_buffer.clear()
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("Gemma unloaded")
