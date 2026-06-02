import numpy as np
from fusion_perception.tracking.cow_points import spawn_points, unpack_cow_outputs

def test_spawn_points_count():
    bbox2d = [100., 50., 300., 200.]   # [x1,y1,x2,y2], area=200×150=30000px²
    pts = spawn_points(bbox2d, beta=0.5, min_pts=8, max_pts=64)
    assert 8 <= len(pts) <= 64

def test_spawn_points_inside_bbox():
    bbox2d = [100., 50., 300., 200.]
    pts = spawn_points(bbox2d)
    for x, y in pts:
        assert 100. <= x <= 300.
        assert 50. <= y <= 200.

def test_spawn_points_minimum():
    tiny_bbox = [100., 50., 102., 52.]  # 2×2 px, very small
    pts = spawn_points(tiny_bbox, min_pts=8, max_pts=64)
    assert len(pts) >= 8   # at least min_pts; grid may round up to next complete square

def test_unpack_cow_outputs():
    import torch
    T, N = 5, 3
    pred_tracks = torch.zeros(1, T, N, 2)
    pred_tracks[0, -1, 0] = torch.tensor([10., 20.])
    pred_tracks[0, -1, 1] = torch.tensor([30., 40.])
    pred_vis = torch.ones(1, T, N)
    track_ids = [1, 2, 3]
    point_counts = [1, 1, 1]
    disps, valids = unpack_cow_outputs(pred_tracks, pred_vis, track_ids, point_counts,
                                       conf_threshold=0.85, min_points=1)
    assert 1 in disps and 2 in disps
    assert valids[1] is True

def test_unpack_cow_outputs_low_confidence():
    import torch
    T, N = 5, 2
    pred_tracks = torch.zeros(1, T, N, 2)
    pred_vis = torch.zeros(1, T, N)  # all visibility = 0 → below threshold
    track_ids = [10, 20]
    point_counts = [1, 1]
    disps, valids = unpack_cow_outputs(pred_tracks, pred_vis, track_ids, point_counts,
                                       conf_threshold=0.85, min_points=1)
    assert valids[10] is False
    assert valids[20] is False
    assert disps[10] == [0.0, 0.0]
