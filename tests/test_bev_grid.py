import numpy as np
from fusion_perception.occupancy.bev_grid import OccupancyBEVGenerator
from fusion_perception.utils.dataclasses import Track

def _make_track(tid, x, z):
    return Track(
        track_id=tid, class_name="car",
        first_seen=0, last_seen=1,
        centroid_history=[[100.0, 100.0]],
        position_3d_history=[[x, 0.0, z]],
        cow_query_point=[100.0, 100.0],
        is_active=True, occlusion_count=0,
    )

def test_occupied_cell_marked():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=1.0,  # no decay for clean test
    )
    tracks = [_make_track(1, x=0.0, z=10.0)]
    grid = gen.update(tracks, frame_idx=0)
    # x=0, z=10 → col=10, row=10
    assert grid.grid[10][10] == 1.0

def test_temporal_decay_reduces_occupancy():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=0.5,
    )
    tracks = [_make_track(1, x=0.0, z=10.0)]
    gen.update(tracks, frame_idx=0)
    grid2 = gen.update([], frame_idx=1)  # no objects — only decay
    assert grid2.grid[10][10] == 0.5


def test_lidar_points_stamp_at_lidar_confidence():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=1.0,
        lidar_confidence=0.6,
    )
    # LiDAR point at x=0, z=5 → col=10, row=5
    lidar_pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    grid = gen.update([], frame_idx=0, lidar_pts=lidar_pts)
    assert abs(grid.grid[5][10] - 0.6) < 1e-5


def test_track_centroid_overwrites_lidar_upward():
    gen = OccupancyBEVGenerator(
        resolution=1.0,
        x_range=[-10.0, 10.0],
        z_range=[0.0, 20.0],
        decay_factor=1.0,
        lidar_confidence=0.6,
    )
    track = Track(
        track_id=1, class_name="car",
        first_seen=0, last_seen=0,
        centroid_history=[[0.0, 0.0]],
        position_3d_history=[[0.0, 0.0, 5.0]],  # x=0, z=5 → col=10, row=5
        cow_query_point=[0.0, 0.0],
        is_active=True, occlusion_count=0,
    )
    lidar_pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    grid = gen.update([track], frame_idx=0, lidar_pts=lidar_pts)
    # Track stamps 1.0 over lidar 0.6
    assert grid.grid[5][10] == 1.0


def test_lidar_does_not_pull_down_prior_frame_track():
    """LiDAR cannot lower a cell value set by a track in a previous frame."""
    from fusion_perception.utils.dataclasses import Track
    gen = OccupancyBEVGenerator(
        resolution=1.0, x_range=[-10.0, 10.0], z_range=[0.0, 20.0],
        decay_factor=1.0,  # no decay so the value stays at 1.0
        lidar_confidence=0.6,
    )
    track = Track(
        track_id=1, class_name="car", first_seen=0, last_seen=0,
        centroid_history=[[0.0, 0.0]],
        position_3d_history=[[0.0, 0.0, 5.0]],
        cow_query_point=[0.0, 0.0], is_active=True, occlusion_count=0,
    )
    # Frame 0: track stamps cell at 1.0
    gen.update([track], frame_idx=0)
    # Frame 1: only LiDAR on same cell — must not reduce 1.0 to 0.6
    lidar_pts = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    grid = gen.update([], frame_idx=1, lidar_pts=lidar_pts)
    assert grid.grid[5][10] == 1.0  # max(1.0, 0.6) = 1.0


def test_lidar_point_outside_grid_is_dropped():
    gen = OccupancyBEVGenerator(
        resolution=1.0, x_range=[-10.0, 10.0], z_range=[0.0, 20.0],
        decay_factor=1.0, lidar_confidence=0.6,
    )
    # Point at x=100 (way outside range)
    lidar_pts = np.array([[100.0, 0.0, 5.0]], dtype=np.float32)
    grid = gen.update([], frame_idx=0, lidar_pts=lidar_pts)
    total = sum(v for row in grid.grid for v in row)
    assert total == 0.0  # nothing was written
