import numpy as np
import pytest

from fusion_perception.segmentation.lidar_classes import (
    CLASS_NAMES, IGNORE, kitti360_to_train,
)
from fusion_perception.segmentation.lidar_probe import confusion_matrix, iou_from_confusion
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.lidar_object import lidar_object_centroids

K = np.array([[700.0, 0, 600.0], [0, 700.0, 180.0], [0, 0, 1]])


def test_kitti360_label_mapping():
    out = kitti360_to_train(np.array([7, 26, 23, 0, -1, 999]))
    assert CLASS_NAMES[out[0]] == "road"
    assert CLASS_NAMES[out[1]] == "car"
    assert list(out[2:]) == [IGNORE] * 4      # sky, unlabeled, license plate, junk


def test_iou_perfect_and_ignore():
    gt = np.array([0, 0, 8, 8, IGNORE], dtype=np.uint8)
    pred = np.array([0, 0, 8, 8, 3], dtype=np.uint8)
    m = iou_from_confusion(confusion_matrix(pred, gt))
    assert m["mIoU"] == pytest.approx(1.0)
    assert m["per_class_iou"]["terrain"] is None


def _det():
    return Detection3D(
        frame_idx=0, class_id=0, class_name="car", score=0.9, score_2d=0.9,
        score_3d=0.9, box_2d=[500, 100, 700, 260], box_3d=[0, 1, 10, 1.8, 1.5, 4, 0],
        centroid_2d=[600, 180], centroid_3d=[0, 1, 10], depth=10.0,
    )


def _scene():
    rng = np.random.default_rng(1)
    car = np.c_[rng.uniform(-0.8, 0.8, 200), rng.uniform(-0.5, 0.5, 200), rng.uniform(9.5, 10.5, 200)]
    wall = np.c_[rng.uniform(-1.5, 1.5, 200), rng.uniform(-1.0, 1.0, 200), np.full(200, 25.0)]
    pts = np.vstack([car, wall])
    labels = np.r_[np.full(200, 8), np.full(200, 4)].astype(np.uint8)
    return pts, labels


def test_lidar_centroid_uses_semantic_car_points():
    pts, labels = _scene()
    out = lidar_object_centroids([_det()], pts, K, labels=labels)
    c = out[0].lidar_centroid_3d
    assert c is not None and abs(c[2] - 10.0) < 0.2


def test_lidar_centroid_without_labels_uses_depth_window():
    pts, _ = _scene()
    c = lidar_object_centroids([_det()], pts, K)[0].lidar_centroid_3d
    assert c is not None and abs(c[2] - 10.0) < 0.3


def test_lidar_centroid_none_without_support():
    out = lidar_object_centroids([_det()], np.zeros((0, 3)), K)
    assert out[0].lidar_centroid_3d is None


def test_read_ply_and_label_transfer(tmp_path):
    pytest.importorskip("scipy")
    from fusion_perception.data.kitti360_semantics import read_ply

    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("semantic", "<i4")])
    arr = np.zeros(3, dtype=dtype)
    arr["x"] = [0, 1, 2]
    arr["semantic"] = [7, 26, 11]
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\nproperty int semantic\nend_header\n"
    )
    p = tmp_path / "w.ply"
    p.write_bytes(header.encode() + arr.tobytes())
    out = read_ply(str(p))
    assert list(out["semantic"]) == [7, 26, 11]
    assert np.allclose(out["x"], [0, 1, 2])
