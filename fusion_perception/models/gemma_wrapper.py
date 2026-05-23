"""Gemma 4 2B reasoning wrapper.

Supports text-only and visual (frame + text) modes via config flag.
Uses bitsandbytes INT4 quantization by default to fit on T4.

Visual mode: annotated frame is passed as PIL image alongside text prompt.
Text mode: prompt only — faster, lower memory, suitable for high-frequency use.

TODO: Add streaming token generation for long summaries.
TODO: Cache KV for repeated prompt prefixes (same system prompt).
"""
from __future__ import annotations
import time
import numpy as np
from PIL import Image
from typing import Optional
from fusion_perception.utils.dataclasses import SceneMemory, ReasoningOutput
from fusion_perception.reasoning.prompt_builder import build_scene_prompt
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("gemma_wrapper")


class GemmaReasoningWrapper:
    """
    Wraps Gemma 4 2B-IT for semantic scene reasoning.
    Call load() once, then reason() per reasoning trigger.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-2b-it",
        quantize_4bit: bool = True,
        max_new_tokens: int = 256,
        device: str = "cuda",
        visual_mode: bool = False,
    ) -> None:
        self.model_id = model_id
        self.quantize_4bit = quantize_4bit
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.visual_mode = visual_mode
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load Gemma model with optional INT4 quantization."""
        logger.info(f"Loading {self.model_id} (4bit={self.quantize_4bit})")
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        kwargs: dict = {"device_map": self.device}

        if self.quantize_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
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
    ) -> ReasoningOutput:
        """
        Run Gemma on current scene memory.
        annotated_frame: RGB [H,W,3] uint8, or None for text-only mode.
        """
        if self._model is None:
            raise RuntimeError("Call load() before reason()")

        use_visual = self.visual_mode and annotated_frame is not None
        prompt_text = build_scene_prompt(memory, fps=fps)

        t0 = time.perf_counter()
        if use_visual:
            raw = self._run_visual(prompt_text, annotated_frame)
        else:
            raw = self._run_text(prompt_text)
        latency_ms = (time.perf_counter() - t0) * 1000

        summary, anomalies, trajectory_nl = self._parse_response(raw)

        logger.info(
            f"Frame {memory.frame_idx} reasoning: {latency_ms:.0f}ms "
            f"({'visual' if use_visual else 'text'})"
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

    def _run_text(self, prompt: str) -> str:
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        return self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _run_visual(self, prompt: str, frame: np.ndarray) -> str:
        import torch
        pil_image = Image.fromarray(frame)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        return self._processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def _parse_response(self, raw: str) -> tuple[str, list[str], str]:
        """Extract summary, anomalies, and trajectory description from raw Gemma output."""
        lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        summary = lines[0] if lines else raw
        anomalies = [l.lstrip("•-* ") for l in lines if l.startswith(("•", "-", "*"))]
        trajectory_nl = " ".join(lines[1:]) if len(lines) > 1 else ""
        return summary, anomalies, trajectory_nl

    def unload(self) -> None:
        import torch
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("Gemma unloaded")
