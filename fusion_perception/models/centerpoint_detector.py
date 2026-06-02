"""CenterPoint LiDAR 3D detector — wraps mmdetection3d for KITTI 3-class model.

Pretrained checkpoint: KITTI 3-class (Car / Pedestrian / Cyclist)
Config  : centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_kitti-3d-3class
Weights : https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/
          centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_kitti-3d-3class/
          centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_kitti-3d-3class_
          20220825_230905-99a75f64.pth

Input  : raw Velodyne [N, 4] float32 (x, y, z, intensity) in LiDAR frame
Output : Detection3D list in camera-rect frame, sorted by score descending
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from fusion_perception.models.base_detector import BaseDetector
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("centerpoint_detector")

_CP_LABEL_TO_CLASS: dict[int, str] = {0: "car", 1: "person", 2: "cyclist"}
_CLASS_TO_ID: dict[str, int] = {
    "car": 0, "person": 1, "cyclist": 2,
    "truck": 3, "bus": 4, "motorcycle": 5,
}

_CKPT_URL = (
    "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/"
    "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_kitti-3d-3class/"
    "centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_kitti-3d-3class"
    "_20220825_230905-99a75f64.pth"
)
_CKPT_NAME = "centerpoint_kitti_3class.pth"
_MODEL_CFG = (
    "centerpoint_pillar02_second_secfpn_head-circlenms_"
    "8xb4-cyclic-20e_kitti-3d-3class"
)


def _get_R_total(calib) -> np.ndarray:
    """3x3 float64 rotation: LiDAR frame -> camera rect frame."""
    from fusion_perception.data.kitti_calibration import Kitti360Calib
    if isinstance(calib, Kitti360Calib):
        # _T already has R_rect baked in for Kitti360Calib
        return calib._T[:3, :3].astype(np.float64)
    # KittiRawCalib: apply R_rect separately
    return (calib._R_rect @ calib._T[:3, :3]).astype(np.float64)


def _lidar_box_to_camera(box7: np.ndarray, calib) -> list[float] | None:
    """Transform [cx,cy,cz,dx,dy,dz,yaw] LiDAR -> [cx,cy,cz,w,h,l,ry] camera.

    Returns None if the box centre is behind the camera (cz_cam <= 0.1).
    """
    cx_l, cy_l, cz_l, dx, dy, dz, yaw = (float(v) for v in box7)
    pt_cam = calib.velo_to_cam(np.array([[cx_l, cy_l, cz_l]], dtype=np.float32))
    if len(pt_cam) == 0:
        return None
    cx_c = float(pt_cam[0, 0])
    cy_c = float(pt_cam[0, 1])
    cz_c = float(pt_cam[0, 2])
    R3 = _get_R_total(calib)
    heading_l = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    heading_c = R3 @ heading_l
    ry = float(np.arctan2(heading_c[0], heading_c[2]))
    # Dimension mapping: dx(forward/depth)->l, dy(lateral)->w, dz(up)->h
    return [cx_c, cy_c, cz_c, float(dy), float(dz), float(dx), ry]


class CenterPointDetector(BaseDetector):
    """mmdetection3d CenterPoint wrapper outputting Detection3D in camera frame."""

    def __init__(
        self,
        score_threshold: float = 0.30,
        ckpt_dir: str = "ckpt",
        device: str = "cuda",
    ) -> None:
        self.score_threshold = score_threshold
        self.ckpt_dir = ckpt_dir
        self.device = device
        self._inferencer = None

    def load(self, checkpoint_path: str = "", device: str = "") -> None:
        import os
        import subprocess
        import sys

        try:
            import mmdet3d  # noqa: F401
            logger.info("mmdetection3d already installed")
        except ImportError:
            logger.info("Installing mmdetection3d ...")
            import torch as _torch
            _cu_tag    = 'cu' + _torch.version.cuda.replace('.', '')
            _torch_tag = 'torch' + '.'.join(_torch.__version__.split('+')[0].split('.')[:2])
            _mmcv_url  = f"https://download.openmmlab.com/mmcv/dist/{_cu_tag}/{_torch_tag}/index.html"
            _ok = subprocess.run(
                [sys.executable, "-m", "pip", "install", "mmcv", "-f", _mmcv_url, "-q"],
                capture_output=True, text=True
            ).returncode == 0
            if not _ok:
                logger.info(f"No CDN wheel for {_cu_tag}/{_torch_tag} — using mmcv-lite")
                subprocess.run([sys.executable, "-m", "pip", "install", "mmcv-lite", "-q"], check=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "mmengine", "mmdet>=3.0.0", "mmdet3d", "-q"],
                check=True,
            )

        from mmdet3d.apis import LidarDet3DInferencer

        ckpt_path = os.path.join(self.ckpt_dir, _CKPT_NAME)
        if not os.path.exists(ckpt_path):
            import urllib.request
            logger.info(f"Downloading CenterPoint KITTI checkpoint -> {ckpt_path}")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            urllib.request.urlretrieve(_CKPT_URL, ckpt_path)

        _dev = device or self.device
        self._inferencer = LidarDet3DInferencer(
            model=_MODEL_CFG,
            weights=ckpt_path,
            device=_dev,
        )
        logger.info(
            f"CenterPoint loaded on {_dev} (score_threshold={self.score_threshold})"
        )

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intrinsics: Optional[np.ndarray],
        prompts: list[str],
        lidar_pts_velo: Optional[np.ndarray] = None,
        calib: object = None,
        sem_mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> list[Detection3D]:
        if self._inferencer is None:
            raise RuntimeError("Call load() before detect()")
        if lidar_pts_velo is None or calib is None or len(lidar_pts_velo) == 0:
            logger.warning(f"Frame {frame_idx}: no LiDAR pts or calib -- skipping")
            return []

        pts = lidar_pts_velo.astype(np.float32)
        if pts.shape[1] == 3:
            pts = np.concatenate(
                [pts, np.zeros((len(pts), 1), dtype=np.float32)], axis=1
            )

        result = self._inferencer(
            {"points": pts},
            batch_size=1,
            return_datasamples=False,
            print_result=False,
        )
        preds = result["predictions"][0]
        boxes  = np.array(preds["bboxes_3d"],  dtype=np.float32)
        labels = np.array(preds["labels_3d"],  dtype=np.int32)
        scores = np.array(preds["scores_3d"],  dtype=np.float32)

        H, W = frame.shape[:2]
        fx  = float(intrinsics[0, 0]) if intrinsics is not None else 552.0
        fy  = float(intrinsics[1, 1]) if intrinsics is not None else 552.0
        ppx = float(intrinsics[0, 2]) if intrinsics is not None else W / 2.0
        ppy = float(intrinsics[1, 2]) if intrinsics is not None else H / 2.0

        detections: list[Detection3D] = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < self.score_threshold:
                continue

            box_cam = _lidar_box_to_camera(boxes[i], calib)
            if box_cam is None:
                continue
            cx_c, cy_c, cz_c, w_c, h_c, l_c, ry = box_cam
            if cz_c < 0.5:
                continue

            u   = float(cx_c / cz_c * fx  + ppx)
            v   = float(cy_c / cz_c * fy  + ppy)
            hw_px = max(4, int(w_c / cz_c * fx / 2.0))
            hh_px = max(4, int(h_c / cz_c * fy / 2.0))
            x1 = float(max(0,     u - hw_px))
            y1 = float(max(0,     v - hh_px))
            x2 = float(min(W - 1, u + hw_px))
            y2 = float(min(H - 1, v + hh_px))

            label      = int(labels[i])
            class_name = _CP_LABEL_TO_CLASS.get(label, "car")
            class_id   = _CLASS_TO_ID.get(class_name, 0)

            detections.append(Detection3D(
                frame_idx=frame_idx,
                class_id=class_id,
                class_name=class_name,
                score=score,
                score_2d=score,
                score_3d=score,
                box_2d=[x1, y1, x2, y2],
                box_3d=[cx_c, cy_c, cz_c, w_c, h_c, l_c, ry],
                centroid_2d=[u, v],
                centroid_3d=[cx_c, cy_c, cz_c],
                depth=cz_c,
            ))

        detections.sort(key=lambda d: d.score, reverse=True)
        logger.debug(f"Frame {frame_idx}: {len(detections)} CenterPoint detections")
        return detections

    def unload(self) -> None:
        import torch
        self._inferencer = None
        torch.cuda.empty_cache()
        logger.info("CenterPointDetector unloaded")
