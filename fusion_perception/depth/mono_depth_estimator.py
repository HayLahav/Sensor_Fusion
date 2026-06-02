"""Monocular metric depth estimator — Depth Anything V2.

Uses the HuggingFace transformers pipeline for Depth Anything V2
(metric outdoor variant).  ViT-Small checkpoint: ~100 MB VRAM, ~30 ms/frame
on T4.

HF model IDs by size:
  small  → depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
  base   → depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf
  large  → depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf

Output: float32 depth map [H, W] in metres, resized to the original frame
resolution.  Values are clipped to max_depth.
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("depth.mono_depth")

_MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
}


class MonoDepthEstimator:
    """Wraps Depth Anything V2 Metric Outdoor for per-frame depth maps.

    Parameters
    ----------
    model_size : {"small", "base", "large"}
        Checkpoint size. "small" is recommended for Colab T4.
    max_depth : float
        Depth values above this (metres) are clamped to max_depth.
    device : str
        "cuda" or "cpu".
    """

    def __init__(
        self,
        model_size: str = "small",
        max_depth: float = 80.0,
        device: str = "cuda",
    ) -> None:
        if model_size not in _MODEL_IDS:
            raise ValueError(f"model_size must be one of {list(_MODEL_IDS)}")
        self.model_id = _MODEL_IDS[model_size]
        self.max_depth = max_depth
        self.device = device
        self._processor = None
        self._model = None

    def load(self) -> None:
        logger.info(f"Loading {self.model_id}")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id
        ).to(self.device)
        self._model.eval()
        log_gpu_memory("MonoDepthEstimator loaded")
        logger.info("MonoDepthEstimator ready")

    def estimate(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Estimate metric depth from an RGB frame.

        Parameters
        ----------
        frame_rgb : np.ndarray
            [H, W, 3] uint8 RGB image.

        Returns
        -------
        np.ndarray | None
            [H, W] float32 depth in metres, resized to input resolution.
            Returns None if the model is not loaded.
        """
        if self._model is None:
            return None

        import torch
        import cv2
        from PIL import Image

        h, w = frame_rgb.shape[:2]
        pil = Image.fromarray(frame_rgb)
        inputs = self._processor(images=pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # predicted_depth: [1, H', W'] in model-internal units (metres for metric model)
        depth_pred = outputs.predicted_depth.squeeze(0).cpu().numpy()  # [H', W']
        depth_pred = np.clip(depth_pred.astype(np.float32), 0.0, self.max_depth)

        # Resize to original resolution
        if depth_pred.shape != (h, w):
            depth_pred = cv2.resize(depth_pred, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth_pred

    def unload(self) -> None:
        if self._model is not None:
            import torch
            self._model.cpu()
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("MonoDepthEstimator unloaded")
