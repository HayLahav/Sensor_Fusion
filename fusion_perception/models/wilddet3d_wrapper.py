"""WildDet3D inference wrapper.

Wraps the allenai/WildDet3D model behind the BaseDetector interface.
Handles FP16 casting, intrinsics estimation, and output parsing.

TODO: Add support for point and box prompts (currently text-only).
TODO: Expose depth map output for downstream occupancy fusion.
"""
from __future__ import annotations
import numpy as np
import torch
from typing import Optional
from fusion_perception.models.base_detector import BaseDetector
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.geometry import box2d_centroid, box3d_centroid, estimate_intrinsics
from fusion_perception.utils.logging_setup import get_logger
from fusion_perception.utils.memory_monitor import log_gpu_memory

logger = get_logger("wilddet3d_wrapper")


class WildDet3DWrapper(BaseDetector):
    """
    Wraps WildDet3D for single-frame 3D detection.

    Input:  RGB frame [H,W,3] uint8 + text prompts
    Output: List[Detection3D] sorted by score descending
    """

    def __init__(self, score_threshold: float = 0.4, fp16: bool = True) -> None:
        self.score_threshold = score_threshold
        self.fp16 = fp16
        self.device = "cpu"
        self._model = None
        self._preprocess = None

    def load(self, checkpoint_path: str, device: str = "cuda") -> None:
        """Load WildDet3D model weights."""
        logger.info(f"Loading WildDet3D from {checkpoint_path} on {device}")
        try:
            from wilddet3d import build_model, preprocess as wpreprocess
        except ImportError:
            raise ImportError(
                "WildDet3D not installed. "
                "Run: pip install git+https://github.com/allenai/WildDet3D"
            )

        self.device = device
        dtype = torch.float16 if self.fp16 else torch.float32
        self._model = build_model(checkpoint_path).to(device).to(dtype)
        self._model.eval()
        self._preprocess = wpreprocess
        log_gpu_memory("WildDet3D loaded")
        logger.info("WildDet3D ready")

    @torch.no_grad()
    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intrinsics: Optional[np.ndarray],
        prompts: list[str],
    ) -> list[Detection3D]:
        """Run WildDet3D on a single frame."""
        if self._model is None:
            raise RuntimeError("Call load() before detect()")

        h, w = frame.shape[:2]
        if intrinsics is None:
            intrinsics = estimate_intrinsics(h, w)

        dtype = torch.float16 if self.fp16 else torch.float32
        data = self._preprocess(
            image=frame,
            intrinsics=intrinsics,
        )

        results = self._model(
            images=data["images"].to(self.device, dtype=dtype),
            intrinsics=data["intrinsics"].to(self.device, dtype=dtype)[None],
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            input_texts=prompts,
        )

        boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids, _ = results
        detections = []

        for i in range(len(scores[0])):
            score = float(scores[0][i])
            if score < self.score_threshold:
                continue

            b2d = boxes_2d[0][i].cpu().tolist()
            b3d = boxes_3d[0][i].cpu().tolist()
            cid = int(class_ids[0][i])

            cx2, cy2 = box2d_centroid(b2d)
            c3 = box3d_centroid(b3d)

            detections.append(Detection3D(
                frame_idx=frame_idx,
                class_id=cid,
                class_name=prompts[cid] if cid < len(prompts) else str(cid),
                score=score,
                score_2d=float(scores_2d[0][i]),
                score_3d=float(scores_3d[0][i]),
                box_2d=b2d,
                box_3d=b3d,
                centroid_2d=[cx2, cy2],
                centroid_3d=c3,
                depth=float(c3[2]),
            ))

        detections.sort(key=lambda d: d.score, reverse=True)
        logger.debug(f"Frame {frame_idx}: {len(detections)} detections")
        return detections

    def unload(self) -> None:
        """Free GPU memory."""
        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("WildDet3D unloaded")
