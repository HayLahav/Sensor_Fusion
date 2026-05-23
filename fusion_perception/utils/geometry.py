"""Geometric utilities: 2D/3D box math, BEV projection, intrinsics."""
import math as _math
import numpy as np


def box2d_centroid(box: list[float]) -> tuple[float, float]:
    """Return pixel centroid [cx, cy] of a [x1,y1,x2,y2] box."""
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def box3d_centroid(box: list[float]) -> list[float]:
    """Return [cx, cy, cz] from a [cx,cy,cz,w,h,l,ry] 3D box."""
    return box[:3]


def camera_to_bev(x_cam: float, z_cam: float) -> tuple[float, float]:
    """
    Project a camera-space (x, z) point into BEV plane.
    Camera convention: z = forward, x = right.
    BEV convention: same — no rotation needed.
    """
    return x_cam, z_cam


def estimate_intrinsics(h: int, w: int) -> np.ndarray:
    """
    Estimate a pinhole intrinsics matrix from image dimensions.
    Uses focal length = max(h, w), principal point at image center.
    Matches WildDet3D's default when no calibration is provided.
    """
    f = float(max(h, w))
    K = np.array([
        [f,   0.0, w / 2.0],
        [0.0, f,   h / 2.0],
        [0.0, 0.0, 1.0    ],
    ], dtype=np.float32)
    return K


def world_to_grid(
    x_cam: float,
    z_cam: float,
    x_range: list[float],
    z_range: list[float],
    resolution: float,
) -> tuple[int, int] | None:
    """
    Convert camera-space (x, z) to BEV grid cell indices (row, col).
    Returns None if the point is outside the grid range.
    """
    if not (x_range[0] <= x_cam <= x_range[1]):
        return None
    if not (z_range[0] <= z_cam <= z_range[1]):
        return None

    col = int((x_cam - x_range[0]) / resolution)
    row = int((z_cam - z_range[0]) / resolution)
    return row, col


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return _math.atan2(_math.sin(angle), _math.cos(angle))


def iou3d(box_a: list[float], box_b: list[float]) -> float:
    """
    Approximate 3D IoU via BEV IoU × height overlap (axis-aligned in world frame).
    box: [cx, cy, cz, theta, l, w, h]  (theta unused — axis-aligned approx)
    l=length(z-axis), w=width(x-axis), h=height(y-axis)
    """
    # BEV (x-z plane): use l and w
    ax1 = box_a[0] - box_a[5] / 2  # cx - w/2
    ax2 = box_a[0] + box_a[5] / 2
    az1 = box_a[2] - box_a[4] / 2  # cz - l/2
    az2 = box_a[2] + box_a[4] / 2

    bx1 = box_b[0] - box_b[5] / 2
    bx2 = box_b[0] + box_b[5] / 2
    bz1 = box_b[2] - box_b[4] / 2
    bz2 = box_b[2] + box_b[4] / 2

    inter_x = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_z = max(0.0, min(az2, bz2) - max(az1, bz1))
    inter_bev = inter_x * inter_z
    area_a = box_a[4] * box_a[5]
    area_b = box_b[4] * box_b[5]
    union_bev = area_a + area_b - inter_bev

    # Height (y-axis)
    ay1 = box_a[1] - box_a[6] / 2
    ay2 = box_a[1] + box_a[6] / 2
    by1 = box_b[1] - box_b[6] / 2
    by2 = box_b[1] + box_b[6] / 2
    h_inter = max(0.0, min(ay2, by2) - max(ay1, by1))
    h_union = max(ay2, by2) - min(ay1, by1)

    iou_bev = inter_bev / (union_bev + 1e-6)
    iou_h = h_inter / (h_union + 1e-6)
    return float(iou_bev * iou_h)
