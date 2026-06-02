from fusion_perception.reasoning.prompt_builder import build_scene_prompt
from fusion_perception.utils.dataclasses import (
    Track, OccupancyGrid, SceneMemory
)

def _make_memory():
    track = Track(
        track_id=1, class_name="car", first_seen=0, last_seen=10,
        centroid_history=[[100.0, 100.0], [110.0, 98.0]],
        position_3d_history=[[2.0, 0.0, 15.0], [2.1, 0.0, 14.0]],
        cow_query_point=[110.0, 98.0],
        is_active=True, occlusion_count=0,
    )
    grid = OccupancyGrid(
        frame_idx=10, resolution=0.5,
        x_range=[-20.0, 20.0], z_range=[0.0, 50.0],
        grid=[[0.0] * 80 for _ in range(100)],
        decay_factor=0.95,
    )
    return SceneMemory(
        frame_idx=10, active_tracks=[track], occupancy_grid=grid,
        event_flags=["new_object:1"], frame_count=10, elapsed_seconds=0.33,
    )

def test_prompt_contains_track_info():
    memory = _make_memory()
    prompt = build_scene_prompt(memory)
    assert "car" in prompt
    assert "Track 1" in prompt

def test_prompt_contains_event_flags():
    memory = _make_memory()
    prompt = build_scene_prompt(memory)
    assert "new_object" in prompt
