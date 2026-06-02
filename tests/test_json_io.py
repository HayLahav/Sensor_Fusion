import json
from pathlib import Path
import tempfile
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.json_io import save_detections, load_detections

def test_save_and_load_detections_roundtrip(tmp_path):
    det = Detection3D(
        frame_idx=0, class_id=0, class_name="car", score=0.9,
        score_2d=0.88, score_3d=0.92,
        box_2d=[10.0, 20.0, 100.0, 80.0],
        box_3d=[2.1, 0.8, 15.3, 1.8, 1.5, 4.2, 0.05],
        centroid_2d=[55.0, 50.0],
        centroid_3d=[2.1, 0.8, 15.3],
        depth=15.3,
    )
    out_path = tmp_path / "detections.json"
    save_detections(
        path=out_path,
        run_id="test_run",
        video_path="clip.mp4",
        prompts=["car"],
        frames_data={0: [det]},
    )
    assert out_path.exists()
    loaded = load_detections(out_path)
    assert loaded["run_id"] == "test_run"
    assert loaded["frames"]["0"][0]["class_name"] == "car"
