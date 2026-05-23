from fusion_perception.utils.dataclasses import (
    Detection3D, Track, OccupancyGrid, SceneMemory, ReasoningOutput,
    GTLabel, BenchmarkResult,
)
import json

def test_detection3d_serializes_to_dict():
    det = Detection3D(
        frame_idx=0, class_id=0, class_name="car", score=0.9,
        score_2d=0.88, score_3d=0.92,
        box_2d=[10.0, 20.0, 100.0, 80.0],
        box_3d=[2.1, 0.8, 15.3, 1.8, 1.5, 4.2, 0.05],
        centroid_2d=[55.0, 50.0],
        centroid_3d=[2.1, 0.8, 15.3],
        depth=15.3,
    )
    d = det.to_dict()
    assert d["class_name"] == "car"
    assert d["depth"] == 15.3

def test_track_json_roundtrip():
    track = Track(
        track_id=1, class_name="car",
        first_seen=0, last_seen=5,
        centroid_history=[[55.0, 50.0], [56.0, 49.0]],
        position_3d_history=[[2.1, 0.8, 15.3]],
        cow_query_point=[55.0, 50.0],
        is_active=True, occlusion_count=0,
    )
    json_str = track.to_json()
    restored = Track.from_json(json_str)
    assert restored.track_id == 1
    assert restored.class_name == "car"

def test_occupancy_grid_shape():
    grid = OccupancyGrid(
        frame_idx=0, resolution=0.5,
        x_range=[-20.0, 20.0], z_range=[0.0, 50.0],
        grid=[[0.0] * 80 for _ in range(100)],
        decay_factor=0.95,
    )
    assert len(grid.grid) == 100
    assert len(grid.grid[0]) == 80

def test_gt_label_serializes():
    label = GTLabel(
        track_id=1,
        class_name="car",
        box_3d=[2.0, 0.5, 15.0, 1.8, 1.5, 4.2, 0.0],
    )
    d = label.to_dict()
    assert d["class_name"] == "car"
    assert d["box_3d"][2] == 15.0

def test_benchmark_result_serializes():
    result = BenchmarkResult(
        dataset="kitti-360",
        log_id="log_0001",
        map=0.42,
        mota=0.61,
        motp=0.38,
        mean_occ_iou=0.55,
        per_class_ap={"car": 0.51, "person": 0.33},
        per_frame_occ_iou=[0.5, 0.6, 0.55],
    )
    d = result.to_dict()
    assert d["map"] == 0.42
    assert d["per_class_ap"]["car"] == 0.51

    restored = BenchmarkResult.from_json(result.to_json())
    assert restored.per_class_ap == {"car": 0.51, "person": 0.33}
    assert restored.per_frame_occ_iou == [0.5, 0.6, 0.55]
