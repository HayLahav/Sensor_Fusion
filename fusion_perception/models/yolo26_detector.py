"""YOLO26n 2D object detector for the hybrid detection pipeline.

Provides accurate class labels for dynamic road users and static traffic
infrastructure. WildDet3D handles metric 3D geometry; this module handles
classification.

COCO class IDs used:
  Dynamic (fed to KF tracker):
    0=person  1=cyclist  2=car  3=motorcyclist  5=bus  7=truck
  Static (fed to SceneMemory event flags, not tracked):
    9=traffic_light  11=stop_sign
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("yolo26_detector")

# Default COCO-ID → pipeline class name mappings
_DYNAMIC_COCO: dict[int, str] = {
    0: "person",
    1: "cyclist",
    2: "car",
    3: "motorcyclist",
    5: "bus",
    7: "truck",
}
_STATIC_COCO: dict[int, str] = {
    9: "traffic_light",
    11: "stop_sign",
}


@dataclass
class YOLODetection:
    """Single 2D detection produced by YOLO26."""
    box_2d: list[float]       # [x1, y1, x2, y2] pixels
    class_id: int             # COCO class ID
    class_name: str           # pipeline class name
    score: float
    centroid_2d: list[float]  # [cx, cy] pixels


class YOLO26Detector:
    """Thin wrapper around YOLO26n for 2D detection and classification.

    In the hybrid pipeline this runs alongside WildDet3D:
      - YOLO26 provides class_name + score (reliable, COCO-trained heads)
      - WildDet3D provides box_3d (metric depth + dimensions)
    The two outputs are fused by 2D IoU in detection_fusion.py.
    """

    def __init__(
        self,
        model_id: str = "yolo26n.pt",
        score_threshold: float = 0.50,
        device: str = "cuda",
        dynamic_coco_ids: dict[int, str] | None = None,
        static_coco_ids: dict[int, str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.score_threshold = score_threshold
        self.device = device
        self.dynamic_coco_ids = dynamic_coco_ids if dynamic_coco_ids is not None else dict(_DYNAMIC_COCO)
        self.static_coco_ids = static_coco_ids if static_coco_ids is not None else dict(_STATIC_COCO)
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.model_id)
        logger.info(f"YOLO26Detector loaded: {self.model_id}")

    def detect(
        self,
        frame_rgb: np.ndarray,
        frame_idx: int,
    ) -> tuple[list[YOLODetection], list[YOLODetection]]:
        """Run YOLO26 on one frame.

        Returns
        -------
        dynamic : list[YOLODetection]
            Road users (person, cyclist, car, motorcyclist, bus, truck).
            Passed to the detection fusion step then the KF tracker.
        static : list[YOLODetection]
            Traffic infrastructure (traffic_light, stop_sign).
            Added to SceneMemory event_flags; NOT tracked.
        """
        if self._model is None:
            raise RuntimeError("Call load() before detect()")

        results = self._model.predict(
            frame_rgb,
            conf=self.score_threshold,
            verbose=False,
            device=self.device,
        )

        dynamic: list[YOLODetection] = []
        static: list[YOLODetection] = []

        boxes_result = results[0].boxes
        if boxes_result is None or len(boxes_result) == 0:
            return dynamic, static

        all_wanted = set(self.dynamic_coco_ids) | set(self.static_coco_ids)

        for i in range(len(boxes_result)):
            cid = int(boxes_result.cls[i].item())
            if cid not in all_wanted:
                continue
            score = float(boxes_result.conf[i].item())
            x1, y1, x2, y2 = [float(v) for v in boxes_result.xyxy[i].tolist()]
            det = YOLODetection(
                box_2d=[x1, y1, x2, y2],
                class_id=cid,
                class_name=self.dynamic_coco_ids.get(cid) or self.static_coco_ids.get(cid, str(cid)),
                score=score,
                centroid_2d=[(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            )
            if cid in self.dynamic_coco_ids:
                dynamic.append(det)
            else:
                static.append(det)

        logger.debug(
            f"Frame {frame_idx}: YOLO26 {len(dynamic)} dynamic  {len(static)} static"
        )
        return dynamic, static

    def unload(self) -> None:
        import torch
        self._model = None
        torch.cuda.empty_cache()
        logger.info("YOLO26Detector unloaded")
