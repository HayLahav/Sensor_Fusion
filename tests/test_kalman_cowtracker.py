import numpy as np
from fusion_perception.tracking.kalman_cowtracker import KalmanCoWTracker
from fusion_perception.utils.dataclasses import Detection3D

K = np.array([[718., 0., 607.], [0., 718., 185.], [0., 0., 1.]], dtype=np.float32)


def _det(cx, cz, score=0.8, frame_idx=0):
    return Detection3D(
        frame_idx=frame_idx, class_id=0, class_name='car',
        score=score, score_2d=score, score_3d=score,
        box_2d=[200., 100., 400., 250.],
        box_3d=[cx, 0., cz, 2., 1.5, 4., 0.],
        centroid_2d=[300., 175.],
        centroid_3d=[cx, 0., cz],
        depth=cz,
    )


def _frame():
    return np.zeros((370, 1224, 3), dtype=np.uint8)


def test_tracker_spawns_track():
    tracker = KalmanCoWTracker(device='cpu')
    tracker.load()
    dets = [_det(0., 15.)]
    tracks = tracker.update(_frame(), dets, 0, intrinsics=K)
    assert len(tracks) == 1
    assert tracks[0].class_name == 'car'


def test_track_persists_across_frames():
    tracker = KalmanCoWTracker(device='cpu', confirm_age=1)
    tracker.load()
    det = _det(0., 15.)
    for i in range(5):
        tracks = tracker.update(_frame(), [det], i, intrinsics=K)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_lost_track_reappears():
    tracker = KalmanCoWTracker(device='cpu', confirm_age=1, lost_patience=10)
    tracker.load()
    for i in range(3):
        tracker.update(_frame(), [_det(0., 15.)], i, intrinsics=K)
    for i in range(3, 6):
        tracker.update(_frame(), [], i, intrinsics=K)
    all_tracks = tracker.get_all_tracks()
    assert any(ts.miss_count > 0 for ts in all_tracks.values())


def test_two_detections_two_tracks():
    tracker = KalmanCoWTracker(device='cpu', confirm_age=1)
    tracker.load()
    dets = [_det(0., 15.), _det(10., 20.)]
    tracks = tracker.update(_frame(), dets, 0, intrinsics=K)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2
