import numpy as np
from fusion_perception.utils.geometry import (
    box2d_centroid, box3d_centroid, camera_to_bev,
    estimate_intrinsics,
)

def test_box2d_centroid():
    cx, cy = box2d_centroid([10.0, 20.0, 100.0, 80.0])
    assert cx == 55.0
    assert cy == 50.0

def test_box3d_centroid():
    xyz = box3d_centroid([2.1, 0.8, 15.3, 1.8, 1.5, 4.2, 0.05])
    assert xyz == [2.1, 0.8, 15.3]

def test_camera_to_bev_forward_maps_to_positive_z():
    bev_x, bev_z = camera_to_bev(x_cam=1.0, z_cam=10.0)
    assert bev_z == 10.0
    assert bev_x == 1.0

def test_estimate_intrinsics_shape():
    K = estimate_intrinsics(h=480, w=640)
    assert K.shape == (3, 3)
    assert K[0, 2] == 320.0  # cx = w/2
    assert K[1, 2] == 240.0  # cy = h/2


from fusion_perception.utils.geometry import iou3d, wrap_angle

def test_iou3d_perfect_overlap():
    box = [0., 0., 10., 0., 4., 2., 1.5]  # [cx,cy,cz,θ,l,w,h]
    assert abs(iou3d(box, box) - 1.0) < 1e-5

def test_iou3d_no_overlap():
    a = [0., 0., 10., 0., 2., 2., 1.5]
    b = [100., 0., 10., 0., 2., 2., 1.5]
    assert iou3d(a, b) == 0.0

def test_iou3d_partial():
    a = [0., 0., 10., 0., 4., 2., 1.5]
    # Shift 1m (half of w=2): produces ~33% BEV overlap. A 2m shift produces
    # edge-to-edge contact with zero overlap — that is a bug in the original spec.
    b = [1., 0., 10., 0., 4., 2., 1.5]
    val = iou3d(a, b)
    assert 0.1 < val < 0.9

def test_wrap_angle_no_change():
    import math
    assert abs(wrap_angle(0.5) - 0.5) < 1e-6

def test_wrap_angle_pi_boundary():
    import math
    # +π and -π are the same angle
    assert abs(wrap_angle(math.pi + 0.1) - (-math.pi + 0.1)) < 1e-5

def test_wrap_angle_minus_pi():
    import math
    result = wrap_angle(-math.pi)
    # atan2 may return +π or -π for this input; both are correct
    assert abs(result - math.pi) < 1e-5 or abs(result + math.pi) < 1e-5

def test_wrap_angle_large_value():
    import math
    # 5.0 rad ≈ 1.57 full rotations; should wrap to 5.0 - 2π ≈ -1.283
    result = wrap_angle(5.0)
    assert abs(result - (5.0 - 2 * math.pi)) < 1e-5
