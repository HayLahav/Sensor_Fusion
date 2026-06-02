"""WildDet3D inference wrapper.

Wraps the allenai/WildDet3D model behind the BaseDetector interface.
Handles FP16 casting, intrinsics estimation, and output parsing.

TODO: Add support for point and box prompts (currently text-only).
TODO: Expose depth map output for downstream occupancy fusion.
"""
from __future__ import annotations
import math as _math
import os
import sys
import numpy as np
import torch
from contextlib import contextmanager
from typing import Optional


@contextmanager
def _suppress_stdout():
    """Redirect stdout to /dev/null for the duration of the block.

    WildDet3D prints verbose NMS debug lines ([NMS CONFIG], [NMS DEBUG])
    on every inference call.  This silences them without touching the model.
    """
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        devnull.close()
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

    def __init__(
        self,
        score_threshold: float = 0.4,
        fp16: bool = True,
        sam3_checkpoint: Optional[str] = None,
    ) -> None:
        self.score_threshold = score_threshold
        self.fp16 = fp16
        self.sam3_checkpoint = sam3_checkpoint  # None → HF auto-download
        self.device = "cpu"
        self._model = None
        self._preprocess = None

    def load(self, checkpoint_path: str, device: str = "cuda") -> None:
        """Load WildDet3D model weights."""
        logger.info(f"Loading WildDet3D from {checkpoint_path} on {device}")
        try:
            from wilddet3d import build_model, preprocess as wpreprocess
        except ImportError as _e:
            raise ImportError(
                f"WildDet3D import failed: {_e}. "
                "Make sure WildDet3D is cloned and its requirements are installed."
            ) from _e

        self.device = device
        # Resolve sam3 checkpoint: user-supplied path > default local path > HF auto-download.
        import os as _os
        sam3_ckpt = self.sam3_checkpoint
        if sam3_ckpt is None:
            _default = "pretrained/sam3/sam3_detector.pt"
            if _os.path.exists(_default):
                sam3_ckpt = _default
                logger.info(f"SAM3 checkpoint found at {_default}")
            else:
                logger.info("SAM3 checkpoint not found locally — will auto-download from HF")
        self._model = build_model(
            checkpoint_path, sam3_checkpoint=sam3_ckpt
        ).to(device)
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
        lidar_pts_velo: np.ndarray | None = None,
        calib: object = None,
    ) -> list[Detection3D]:
        """Run WildDet3D on a single frame."""
        if self._model is None:
            raise RuntimeError("Call load() before detect()")

        h, w = frame.shape[:2]
        if intrinsics is None:
            intrinsics = estimate_intrinsics(h, w)

        data = self._preprocess(
            image=frame,
            intrinsics=intrinsics,
        )

        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16), \
             _suppress_stdout():
            results = self._model(
                images=data["images"].to(self.device),
                intrinsics=data["intrinsics"].to(self.device)[None],
                input_hw=[data["input_hw"]],
                original_hw=[data["original_hw"]],
                padding=[data["padding"]],
                input_texts=prompts,
            )

        boxes_2d, boxes_3d, scores, scores_2d, scores_3d, class_ids, _ = results
        detections = []

        for i in range(len(scores[0])):
            score = min(1.0, max(0.0, float(scores[0][i])))
            if score < self.score_threshold:
                continue

            b2d = boxes_2d[0][i].cpu().tolist()
            b3d_raw = boxes_3d[0][i].cpu().tolist()
            cid = int(class_ids[0][i])

            # WildDet3D native: [cx, cy, cz, h, l, w, qw, qx, qy, qz] (10 elem)
            # or older 7-elem:  [cx, cy, cz, h, l, w, ry]
            # Detection3D wants: [cx, cy, cz, w, h, l, ry]
            #   w = x-extent (lateral), h = y-extent (height), l = z-extent (depth)
            if len(b3d_raw) >= 10:
                h_b, l_b, w_b = b3d_raw[3], b3d_raw[4], b3d_raw[5]
                # Extract yaw from quaternion (rotation around camera y-axis)
                ry = 2.0 * _math.atan2(float(b3d_raw[8]), float(b3d_raw[6]))
            else:
                h_b, l_b, w_b = b3d_raw[3], b3d_raw[4], b3d_raw[5]
                ry = float(b3d_raw[6])
            b3d = [b3d_raw[0], b3d_raw[1], b3d_raw[2], w_b, h_b, l_b, ry]

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
