import numpy as np
from fusion_perception.utils.video_loader import VideoLoader

def test_video_loader_yields_frames_with_metadata(tmp_path):
    import cv2
    vpath = str(tmp_path / "test.mp4")
    out = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(5):
        out.write(np.zeros((48, 64, 3), dtype=np.uint8))
    out.release()

    loader = VideoLoader(source=vpath, resize_hw=(48, 64), max_frames=None)
    frames = list(loader)
    assert len(frames) == 5
    idx, frame, meta = frames[0]
    assert idx == 0
    assert frame.shape == (48, 64, 3)
    assert "fps" in meta
    assert "total_frames" in meta

def test_video_loader_respects_max_frames(tmp_path):
    import cv2
    vpath = str(tmp_path / "test2.mp4")
    out = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for _ in range(10):
        out.write(np.zeros((48, 64, 3), dtype=np.uint8))
    out.release()

    loader = VideoLoader(source=vpath, resize_hw=None, max_frames=3)
    frames = list(loader)
    assert len(frames) == 3
