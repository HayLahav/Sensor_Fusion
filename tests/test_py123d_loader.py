from unittest.mock import MagicMock, patch
import numpy as np
from fusion_perception.data.py123d_loader import Py123dLoader
from fusion_perception.utils.dataclasses import GTLabel


def _make_mock_scene():
    """Build a minimal mock of py123d SceneAPI."""
    scene = MagicMock()
    scene.fps = 10.0

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    scene.cameras.__getitem__.return_value.frames = [
        (0.0, fake_frame),
        (0.1, fake_frame),
        (0.2, fake_frame),
    ]
    scene.cameras.__getitem__.return_value.intrinsics = np.eye(3, dtype=np.float32)
    scene.get_lidar_at_timestamp.return_value = np.zeros((100, 3), dtype=np.float32)
    scene.get_labels_at_timestamp.return_value = [
        {"track_id": 1, "class_name": "car",
         "box_3d": [2.0, 0.5, 15.0, 1.8, 1.5, 4.2, 0.0]},
    ]
    return scene


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_loader_yields_frames(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    frames = list(loader)
    assert len(frames) == 3
    idx, frame, meta = frames[0]
    assert idx == 0
    assert frame.shape == (480, 640, 3)
    assert "fps" in meta
    assert "total_frames" in meta


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_intrinsics_returns_3x3(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    K = loader.get_intrinsics()
    assert K.shape == (3, 3)
    assert K.dtype == np.float32


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_lidar_returns_nx3(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    pts = loader.get_lidar(frame_idx=0)
    assert pts is not None
    assert pts.ndim == 2
    assert pts.shape[1] == 3


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_gt_labels_returns_list(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    labels = loader.get_gt_labels(frame_idx=0)
    assert isinstance(labels, list)
    assert all(isinstance(lb, GTLabel) for lb in labels)
    assert labels[0].class_name == "car"


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_meta_dict_is_not_shared(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    frames = list(loader)
    _, _, meta0 = frames[0]
    _, _, meta1 = frames[1]
    meta0["injected"] = True
    assert "injected" not in meta1  # separate dicts


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_meta_contains_timestamp(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    _, _, meta = next(iter(loader))
    assert "timestamp" in meta


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_lidar_returns_none_out_of_bounds(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    assert loader.get_lidar(999) is None
    assert loader.get_lidar(-1) is None


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_gt_labels_returns_empty_out_of_bounds(mock_api_cls):
    mock_api_cls.return_value = _make_mock_scene()
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    assert loader.get_gt_labels(999) == []
    assert loader.get_gt_labels(-1) == []


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_lidar_returns_none_on_exception(mock_api_cls):
    scene = _make_mock_scene()
    scene.get_lidar_at_timestamp.side_effect = RuntimeError("lidar unavailable")
    mock_api_cls.return_value = scene
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    result = loader.get_lidar(0)
    assert result is None


@patch("fusion_perception.data.py123d_loader.SceneAPI")
def test_get_gt_labels_returns_empty_on_exception(mock_api_cls):
    scene = _make_mock_scene()
    scene.get_labels_at_timestamp.side_effect = RuntimeError("labels unavailable")
    mock_api_cls.return_value = scene
    loader = Py123dLoader(log_dir="fake/log", camera_name="camera")
    result = loader.get_gt_labels(0)
    assert result == []
