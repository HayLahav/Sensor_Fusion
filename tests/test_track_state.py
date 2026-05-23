from fusion_perception.tracking.track_state import TrackState, TrackStatus
import numpy as np

def test_track_status_values():
    assert TrackStatus.TENTATIVE != TrackStatus.CONFIRMED
    assert TrackStatus.CONFIRMED != TrackStatus.LOST
    assert TrackStatus.TENTATIVE != TrackStatus.LOST

def test_track_state_defaults():
    ts = TrackState(track_id=1, class_name="car", kf=None, status=TrackStatus.TENTATIVE)
    assert ts.age == 0
    assert ts.miss_count == 0
    assert ts.cow_points_abs is None
    assert len(ts.centroid_history) == 0
    assert len(ts.position_3d_history) == 0

def test_track_state_to_track():
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=10, dim_z=7)
    kf.x = np.zeros((10, 1))
    kf.x[:7, 0] = [1., 0., 15., 0., 4., 2., 1.5]
    ts = TrackState(
        track_id=5, class_name="car", kf=kf,
        status=TrackStatus.CONFIRMED,
        first_seen=0, last_seen=10,
        centroid_history=[[320., 240.]],
        position_3d_history=[[1., 0., 15.]],
    )
    track = ts.to_track()
    assert track.track_id == 5
    assert track.class_name == "car"
    assert track.is_active is True

def test_to_track_raises_with_no_kf():
    import pytest
    ts = TrackState(track_id=2, class_name="car", kf=None, status=TrackStatus.CONFIRMED)
    with pytest.raises(ValueError, match="kf=None"):
        ts.to_track()

def test_to_track_cow_query_point_fallback():
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=10, dim_z=7)
    kf.x = np.zeros((10, 1))
    ts = TrackState(track_id=3, class_name="car", kf=kf, status=TrackStatus.CONFIRMED)
    track = ts.to_track()
    assert track.cow_query_point == [0., 0.]

def test_to_track_cow_query_point_serializable():
    from filterpy.kalman import KalmanFilter
    kf = KalmanFilter(dim_x=10, dim_z=7)
    kf.x = np.zeros((10, 1))
    ts = TrackState(
        track_id=7, class_name="car", kf=kf, status=TrackStatus.CONFIRMED,
        cow_points_abs=np.array([[320., 240.], [310., 235.]]),
    )
    track = ts.to_track()
    assert track.cow_query_point == [320., 240.]
    # Must be native Python floats for JSON serialization
    assert all(isinstance(v, float) for v in track.cow_query_point)
