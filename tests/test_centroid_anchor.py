from fusion_perception.tracking.centroid_anchor import (
    match_detections_to_tracks, assign_new_track_id
)
from fusion_perception.utils.dataclasses import Detection3D, Track

def _make_det(frame_idx, cx, cy, cz=15.0):
    return Detection3D(
        frame_idx=frame_idx, class_id=0, class_name="car",
        score=0.9, score_2d=0.88, score_3d=0.92,
        box_2d=[cx-20, cy-20, cx+20, cy+20],
        box_3d=[0.0, 0.0, cz, 1.8, 1.5, 4.2, 0.0],
        centroid_2d=[cx, cy], centroid_3d=[0.0, 0.0, cz], depth=cz,
    )

def _make_track(tid, cx, cy):
    return Track(
        track_id=tid, class_name="car",
        first_seen=0, last_seen=0,
        centroid_history=[[cx, cy]],
        position_3d_history=[[0.0, 0.0, 15.0]],
        cow_query_point=[cx, cy],
        is_active=True, occlusion_count=0,
    )

def test_nearby_detection_matches_existing_track():
    tracks = {1: _make_track(1, 100.0, 100.0)}
    dets = [_make_det(1, 102.0, 101.0)]   # 2.2px away — well within threshold
    matched, unmatched_dets, unmatched_tracks = match_detections_to_tracks(
        detections=dets, active_tracks=tracks, nn_threshold=50.0
    )
    assert (dets[0], 1) in matched
    assert len(unmatched_dets) == 0

def test_far_detection_creates_new_track():
    tracks = {1: _make_track(1, 100.0, 100.0)}
    dets = [_make_det(1, 500.0, 400.0)]   # far away
    matched, unmatched_dets, unmatched_tracks = match_detections_to_tracks(
        detections=dets, active_tracks=tracks, nn_threshold=50.0
    )
    assert len(matched) == 0
    assert len(unmatched_dets) == 1

def test_assign_new_track_id_increments():
    assert assign_new_track_id({}) == 1
    assert assign_new_track_id({1: None, 2: None}) == 3
