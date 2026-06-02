"""Relabel CenterPoint 3D detections using YOLO26n-sem Cityscapes class mask.

For each CenterPoint Detection3D box, project the centroid footprint through K
into the semantic mask.  The dominant Cityscapes vehicle/person class inside
the projected region overrides the CenterPoint class label (car/ped/cyclist)
with a richer label (truck, bus, motorcycle, bicycle, rider …).

Non-vehicle regions (road, building, vegetation …) are ignored; the
CenterPoint label is kept as fallback.
"""
from __future__ import annotations
import copy
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("detection_fusion")

# Cityscapes trainId → Detection3D class name  (vehicle / person classes only)
_SEM_TO_CLASS: dict[int, str] = {
    11: "person",
    12: "cyclist",      # rider
    13: "car",
    14: "truck",
    15: "bus",
    16: "truck",        # train → truck (closest category)
    17: "motorcycle",
    18: "cyclist",      # bicycle
}

_CLASS_TO_ID: dict[str, int] = {
    "car": 0, "person": 1, "cyclist": 2,
    "truck": 3, "bus": 4, "motorcycle": 5,
}


def _dominant_sem_class(
    box_3d: list[float],
    intrinsics: np.ndarray,
    sem_mask: np.ndarray,
    frame_hw: tuple[int, int],
) -> int | None:
    """Return dominant Cityscapes vehicle trainId inside the projected 2D box.

    Returns None when the box is behind the camera, projects off-screen,
    or no vehicle-class pixels are found in the region.
    """
    cx, cy, cz, w, h, l, ry = box_3d
    if cz < 0.5:
        return None
    H_img, W_img = frame_hw
    fx  = float(intrinsics[0, 0])
    fy  = float(intrinsics[1, 1])
    ppx = float(intrinsics[0, 2])
    ppy = float(intrinsics[1, 2])

    u_c   = int(cx / cz * fx + ppx)
    v_c   = int(cy / cz * fy + ppy)
    hw_px = max(4, int(w / cz * fx / 2.0))
    hh_px = max(4, int(h / cz * fy / 2.0))
    x1 = max(0,       u_c - hw_px)
    y1 = max(0,       v_c - hh_px)
    x2 = min(W_img - 1, u_c + hw_px)
    y2 = min(H_img - 1, v_c + hh_px)
    if x2 <= x1 or y2 <= y1:
        return None

    patch = sem_mask[y1:y2, x1:x2].ravel()
    vehicle_ids = np.array(list(_SEM_TO_CLASS.keys()), dtype=np.int16)
    vehicle_px  = patch[np.isin(patch, vehicle_ids)]
    if len(vehicle_px) == 0:
        return None

    vals, counts = np.unique(vehicle_px, return_counts=True)
    return int(vals[counts.argmax()])


def fuse_lidar_with_sem(
    cp_detections: list[Detection3D],
    sem_mask: np.ndarray | None,
    intrinsics: np.ndarray | None,
    frame_hw: tuple[int, int],
) -> list[Detection3D]:
    """Relabel CenterPoint detections with Cityscapes semantic class.

    Parameters
    ----------
    cp_detections : Detection3D list from CenterPointDetector (car/ped/cyclist).
    sem_mask      : int16 H×W Cityscapes trainId map from RoadSegmentor.segment().
                    Pass None to skip relabeling (CenterPoint labels kept).
    intrinsics    : [3,3] float32 camera K matrix.
    frame_hw      : (H, W) of the camera frame.

    Returns
    -------
    list[Detection3D] with updated class_name/class_id where sem vote wins,
    sorted by score descending.
    """
    if not cp_detections:
        return []
    if sem_mask is None or intrinsics is None:
        return cp_detections

    relabeled: list[Detection3D] = []
    n_changed = 0

    for det in cp_detections:
        dom = _dominant_sem_class(det.box_3d, intrinsics, sem_mask, frame_hw)
        if dom is not None and dom in _SEM_TO_CLASS:
            new_class = _SEM_TO_CLASS[dom]
            new_id    = _CLASS_TO_ID.get(new_class, det.class_id)
            det2 = copy.copy(det)
            det2.class_name = new_class
            det2.class_id   = new_id
            relabeled.append(det2)
            if new_class != det.class_name:
                n_changed += 1
                logger.debug(f"  relabeled {det.class_name}→{new_class} (sem={dom})")
        else:
            relabeled.append(det)

    logger.debug(
        f"fuse_lidar_with_sem: {n_changed}/{len(cp_detections)} boxes relabeled"
    )
    relabeled.sort(key=lambda d: d.score, reverse=True)
    return relabeled
