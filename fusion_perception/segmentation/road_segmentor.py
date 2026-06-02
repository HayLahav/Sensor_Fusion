"""Road / drivable-area segmentation using YOLO26 semantic (Cityscapes).

Returns a full int H×W class map (Cityscapes trainIds). Callers use it
both for road filtering (classes 0/1/9) and for semantic BEV coloring.

Cityscapes trainId classes relevant to driving:
  0=road  1=sidewalk  8=vegetation  9=terrain
  11=person  12=rider  13=car  14=truck  15=bus  17=motorcycle  18=bicycle
"""
from __future__ import annotations
import numpy as np


_DEFAULT_MODEL = "yolo26n-sem.pt"   # nano: 1.6M params, 78.3 mIoU, 4.4ms on 1024×2048
_ROAD_CLASSES = frozenset([0, 1, 9])


class RoadSegmentor:
    """Thin wrapper around YOLO26n-sem returning a full Cityscapes class map."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        road_classes: frozenset[int] = _ROAD_CLASSES,
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.road_classes = road_classes
        self.device = device
        self._model = None
        self._warned_none = False

    def load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.model_id)

    def segment(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return int H×W Cityscapes class map (-1 where model has no prediction)."""
        import cv2
        H, W = frame_rgb.shape[:2]

        results = self._model.predict(
            frame_rgb,
            verbose=False,
            device=self.device,
        )

        seg = results[0]

        # YOLO26n-sem outputs a semantic class map in seg.semantic_mask
        # (not seg.masks, which is for instance segmentation).
        pred_small: np.ndarray | None = None

        if hasattr(seg, 'semantic_mask') and seg.semantic_mask is not None:
            sm = seg.semantic_mask
            # SemanticMask is an Ultralytics wrapper; raw tensor is in .data
            raw = sm.data if hasattr(sm, 'data') else sm
            if hasattr(raw, 'cpu'):
                pred_small = raw.cpu().numpy().astype(np.int32)
            else:
                pred_small = np.asarray(raw, dtype=np.int32)

        elif seg.masks is not None:
            mask_data = seg.masks.data
            if mask_data.ndim == 2:
                pred_small = mask_data.cpu().numpy().astype(np.int32)
            elif mask_data.ndim == 3 and mask_data.shape[0] > 1:
                pred_small = mask_data.argmax(0).cpu().numpy().astype(np.int32)
            else:
                pred_small = mask_data.squeeze(0).cpu().numpy().astype(np.int32)

        if pred_small is None:
            return np.full((H, W), -1, dtype=np.int16)

        if pred_small.shape != (H, W):
            pred_small = cv2.resize(
                pred_small.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

        return pred_small.astype(np.int16)

    def road_mask(self, class_map: np.ndarray) -> np.ndarray:
        """Derive bool road mask from a class map returned by segment()."""
        mask = np.zeros(class_map.shape, dtype=bool)
        for cls in self.road_classes:
            mask |= (class_map == cls)
        return mask

    def unload(self) -> None:
        import torch
        self._model = None
        torch.cuda.empty_cache()
