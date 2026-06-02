"""Project WildDet3D 3D bounding boxes onto a 2D image frame.

box_3d format (10 elements from WildDet3D):
  [cx, cy, cz, h, l, w, qw, qx, qy, qz]
  cx/cy/cz : centre in rectified camera frame (metres); x right, y down, z forward
  h        : height  (y-axis extent)
  l        : length  (x-axis extent in canonical frame; KITTI ry=0 → object faces +x)
  w        : width   (z-axis extent in canonical frame)
  qw..qz   : quaternion rotation (object → camera frame)
"""
from __future__ import annotations
import cv2
import numpy as np

# 12 wireframe edges: top face + bottom face + 4 vertical pillars
_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array([
        [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)    ],
        [2*(qx*qy + qz*qw),       1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)    ],
        [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float32)


def box3d_corners(box3d: list[float]) -> np.ndarray:
    """Compute [8, 3] corner coordinates in camera space."""
    cx, cy, cz = box3d[0], box3d[1], box3d[2]
    h, l, w = box3d[3], box3d[4], box3d[5]
    hh, hl, hw = h / 2, l / 2, w / 2
    # KITTI convention: length (l) along x-axis so ry=0→faces+x, ry=-π/2→faces+z(forward)
    # y=0 at top of box (y is down in camera frame)
    corners = np.array([
        [ hl, -hh,  hw],
        [-hl, -hh,  hw],
        [-hl, -hh, -hw],
        [ hl, -hh, -hw],
        [ hl,  hh,  hw],
        [-hl,  hh,  hw],
        [-hl,  hh, -hw],
        [ hl,  hh, -hw],
    ], dtype=np.float32)
    if len(box3d) >= 10:
        R = _quat_to_rot(box3d[6], box3d[7], box3d[8], box3d[9])
        corners = (R @ corners.T).T
    corners += np.array([cx, cy, cz], dtype=np.float32)
    return corners


def project_box3d(
    frame: np.ndarray,
    box3d: list[float],
    K: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a 3D bounding box wireframe onto frame (in-place)."""
    # Skip objects closer than 2 m — projection becomes degenerate and huge
    if box3d[2] < 2.0:
        return

    corners = box3d_corners(box3d)
    if np.any(corners[:, 2] <= 0.1):
        return

    pts_h = (K.astype(np.float32) @ corners.T).T   # [8, 3]
    pts2d = (pts_h[:, :2] / pts_h[:, 2:3]).astype(np.int32)  # [8, 2]

    h_img, w_img = frame.shape[:2]

    # Skip if projected 2D bounding box is more than 2× the image size
    x_min, x_max = int(pts2d[:, 0].min()), int(pts2d[:, 0].max())
    y_min, y_max = int(pts2d[:, 1].min()), int(pts2d[:, 1].max())
    if (x_max - x_min) > w_img * 2 or (y_max - y_min) > h_img * 2:
        return

    for i, j in _EDGES:
        p1, p2 = tuple(pts2d[i]), tuple(pts2d[j])
        if (abs(p1[0]) < w_img * 1.5 and abs(p1[1]) < h_img * 1.5 and
                abs(p2[0]) < w_img * 1.5 and abs(p2[1]) < h_img * 1.5):
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
