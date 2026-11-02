"""Find each detection's own LiDAR points and their median 3D position.

Points are those that project inside the detection's 2D box. When per-point
semantic labels are available (Utonia linear probe), only points labelled
with the detection's class are kept, which removes the ground, walls and
background that also fall inside the box. Without labels, a depth window
around the detection's depth and a cut of the bottom of the box are used.

The result (Detection3D.lidar_centroid_3d) is used by the tracker's motion
gate to tell parked cars from moving ones.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Optional
import numpy as np

from fusion_perception.segmentation.lidar_classes import DETECTION_CLASS_TO_LIDAR
from fusion_perception.utils.dataclasses import Detection3D


def lidar_object_centroids(
    detections: list[Detection3D],
    lidar_pts_cam: Optional[np.ndarray],
    intrinsics: Optional[np.ndarray],
    labels: Optional[np.ndarray] = None,
    min_points: int = 8,
    box_shrink: float = 0.1,
    depth_window: float = 2.0,
    bottom_cut: float = 0.15,
) -> list[Detection3D]:
    """Return detections with lidar_centroid_3d set where LiDAR support exists.

    lidar_pts_cam : [N,3] rectified camera frame (x right, y down, z forward).
    labels        : optional [N] training-class ids aligned with lidar_pts_cam.
    """
    if not detections or lidar_pts_cam is None or intrinsics is None or len(lidar_pts_cam) == 0:
        return detections

    fwd = lidar_pts_cam[:, 2] > 0.5
    pts = lidar_pts_cam[fwd]
    lab = labels[fwd] if labels is not None else None
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * pts[:, 0] / pts[:, 2] + cx
    v = fy * pts[:, 1] / pts[:, 2] + cy

    out: list[Detection3D] = []
    for det in detections:
        x1, y1, x2, y2 = det.box_2d
        bw, bh = x2 - x1, y2 - y1
        sx, sy = bw * box_shrink / 2, bh * box_shrink / 2
        in_box = (u >= x1 + sx) & (u <= x2 - sx) & (v >= y1 + sy) & (v <= y2 - sy)

        wanted = DETECTION_CLASS_TO_LIDAR.get(det.class_name.lower())
        if lab is not None and wanted is not None:
            mask = in_box & np.isin(lab, list(wanted))
        else:
            # No semantics: keep points near the detection depth and above the
            # bottom strip of the box (where ground returns usually land).
            near = np.abs(pts[:, 2] - det.depth) < depth_window
            above_ground = v <= y2 - bh * bottom_cut
            mask = in_box & near & above_ground

        if int(mask.sum()) >= min_points:
            c = np.median(pts[mask], axis=0)
            out.append(replace(det, lidar_centroid_3d=[float(c[0]), float(c[1]), float(c[2])]))
        else:
            out.append(det)
    return out
