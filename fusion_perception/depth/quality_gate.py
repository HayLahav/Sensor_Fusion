"""Camera frame quality gate for depth estimation.

Decides whether the current frame is reliable enough for monocular depth
estimation. Two modes:

  fast (default):
    Laplacian variance (sharpness) + mean luminance check. Zero VRAM overhead.
    Returns a score in [0, 1]; below min_score → skip depth this frame.

  clip_iqa:
    CLIP-IQA: uses CLIP text–image similarity against "Good photo" / "Bad photo"
    antonym pairs. More accurate but loads ~330 MB into VRAM.
    Requires: pip install transformers torch
"""
from __future__ import annotations
import numpy as np
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("depth.quality_gate")

_CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
_ANTONYMS = [("Good photo", "Bad photo"), ("Sharp image", "Blurry image")]


class DepthQualityGate:
    """Gate that scores each frame; returns True when depth is usable.

    Parameters
    ----------
    use_clip_iqa : bool
        True → load CLIP and compute CLIP-IQA score.
        False → fast Laplacian + luminance heuristic (default).
    min_score : float
        Frames scoring below this threshold are rejected (depth skipped).
    device : str
        Device for CLIP model when use_clip_iqa=True.
    """

    def __init__(
        self,
        use_clip_iqa: bool = False,
        min_score: float = 0.3,
        device: str = "cuda",
    ) -> None:
        self.use_clip_iqa = use_clip_iqa
        self.min_score = min_score
        self.device = device
        self._clip_model = None
        self._clip_processor = None

    def load(self) -> None:
        if not self.use_clip_iqa:
            logger.info("DepthQualityGate: fast mode (Laplacian + luminance)")
            return
        logger.info(f"DepthQualityGate: loading CLIP-IQA ({_CLIP_MODEL_ID})")
        from transformers import CLIPModel, CLIPProcessor
        self._clip_model = CLIPModel.from_pretrained(_CLIP_MODEL_ID).to(self.device)
        self._clip_model.eval()
        self._clip_processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_ID)
        logger.info("CLIP-IQA model ready")

    def unload(self) -> None:
        if self._clip_model is not None:
            import torch
            self._clip_model.cpu()
            del self._clip_model
            self._clip_model = None
            torch.cuda.empty_cache()

    def score(self, frame_rgb: np.ndarray) -> float:
        """Return a quality score in [0, 1]. Higher = better quality."""
        if self.use_clip_iqa and self._clip_model is not None:
            return self._clip_iqa_score(frame_rgb)
        return self._fast_score(frame_rgb)

    def is_usable(self, frame_rgb: np.ndarray) -> bool:
        return self.score(frame_rgb) >= self.min_score

    # ------------------------------------------------------------------
    def _fast_score(self, frame_rgb: np.ndarray) -> float:
        import cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        # Sharpness: Laplacian variance (0 = completely blurry)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = float(np.clip(lap_var / 500.0, 0.0, 1.0))  # 500 → saturates score
        # Exposure: penalise very dark (<30 mean) or very bright (>220 mean) frames
        mean_lum = float(gray.mean())
        if mean_lum < 30.0:
            exposure = mean_lum / 30.0
        elif mean_lum > 220.0:
            exposure = (255.0 - mean_lum) / 35.0
        else:
            exposure = 1.0
        exposure = float(np.clip(exposure, 0.0, 1.0))
        score = 0.6 * sharpness + 0.4 * exposure
        logger.debug(f"Fast quality: sharpness={sharpness:.2f} exposure={exposure:.2f} → {score:.2f}")
        return score

    def _clip_iqa_score(self, frame_rgb: np.ndarray) -> float:
        import torch
        from PIL import Image
        pil = Image.fromarray(frame_rgb)
        scores: list[float] = []
        for pos, neg in _ANTONYMS:
            inputs = self._clip_processor(
                text=[pos, neg], images=pil, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                logits = self._clip_model(**inputs).logits_per_image[0]
            probs = logits.softmax(dim=0)
            scores.append(float(probs[0]))   # probability of the positive prompt
        return float(np.mean(scores))
