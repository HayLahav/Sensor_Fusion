"""Build Qwen-Drive planning inputs from this pipeline's ego state and camera views.

Two outputs:
  scene_record()  one JSON line in Qwen-Drive's scene-file format (docs/data.md),
                  for running Qwen-Drive's own scripts/run_planning.py in its
                  separate environment (it pins torch 2.8 / transformers 5.x,
                  this project pins torch 2.5.1).
  driving_scene() a qwen_drive.DrivingScene for running the model in-process
                  when both live in one environment.

Camera timestamps: 4 frames at 2 Hz (t-1.5, -1.0, -0.5, 0 s), grouped by view
(front, front-left, front-right), as the model was trained.
"""
from __future__ import annotations
from typing import Any, Optional
import numpy as np

from fusion_perception.planning.ego_state import EgoState
from fusion_perception.planning.kitti360_views import VIEW_TAGS

CAMERA_TIME_OFFSETS_S = (-1.5, -1.0, -0.5, 0.0)
_TIME_LABELS = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")
_NAV_TEXT = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")
_CAM_ORDER = ("FRONT", "FRONT_LEFT", "FRONT_RIGHT")

# Same text Qwen-Drive builds in scene.DrivingScene.instruction() (Apache-2.0),
# so a scene file and an in-process scene give the model the same prompt.
_INSTRUCTION_HEADER = (
    "The input images are organized by camera view. Each view contains {num_frames} temporal "
    "frames captured at {interval:g}s intervals (frame 0 at {first_label}, frame {last_frame} is "
    "the current frame at t=0s).\n"
    "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. "
    "Positive x points forward, positive y points left, and a positive heading indicates a left "
    "turn：\n"
)


def camera_frame_offsets(fps: float) -> tuple[int, ...]:
    """Frame offsets for the 4 camera timestamps (KITTI-360 10 Hz → -15, -10, -5, 0)."""
    return tuple(int(round(t * fps)) for t in CAMERA_TIME_OFFSETS_S)


def instruction_text(history: np.ndarray, nav_command: int, num_frames: int = 4) -> str:
    history = np.asarray(history, dtype=np.float64) + 0.0   # no "-0.0000" in the prompt
    labels = _TIME_LABELS[-num_frames:]
    stride = max(1, (len(history) - 1) // max(1, num_frames - 1))
    lines = "".join(
        " -{}: ({:.4f}, {:.4f}, {:.4f});\n".format(label, *history[i * stride])
        for i, label in enumerate(labels)
    )
    header = _INSTRUCTION_HEADER.format(
        num_frames=num_frames, interval=1.5 / max(1, num_frames - 1),
        first_label=labels[0], last_frame=num_frames - 1,
    )
    return f"{header}{lines}2. Active navigation command: [{_NAV_TEXT[int(nav_command)]}]"


def scene_token(sequence: str, frame_idx: int) -> str:
    return f"{sequence}_{frame_idx:010d}"


def scene_record(
    sequence: str,
    state: EgoState,
    image_paths: dict[str, list[str]],
) -> dict:
    """One Qwen-Drive scene-file record.

    image_paths: {view tag: [4 paths, oldest first]}, relative to the frame
    root passed to run_planning.py as --image-root.
    """
    content: list[dict[str, Any]] = []
    for tag in VIEW_TAGS:
        paths = image_paths[tag]
        content.append({"text": tag})
        for k, p in enumerate(paths):
            content.append({"text": f"frame: {k}"})
            content.append({"image": p, "timestamp_idx": k,
                            "timestamp_offset_sec": CAMERA_TIME_OFFSETS_S[-len(paths):][k]})
    content.append({"text": instruction_text(np.asarray(state.history), state.nav_command,
                                             num_frames=len(image_paths[VIEW_TAGS[0]]))})
    future = state.future
    return {
        "type": "chatml",
        "messages": [{"role": "user", "content": content},
                     {"role": "assistant", "content": []}],
        "meta_info": {
            "dataset": "KITTI-360",
            "token": scene_token(sequence, state.frame_idx),
            "scene_token": sequence,
            "frame_idx": state.frame_idx,
            "cam_order": list(_CAM_ORDER),
        },
        "trajectory": {
            "hist_traj_10hz": state.history,
            "hist_vel_10hz": state.history_velocity,
            "hist_acc_10hz": state.history_acceleration,
            "future_traj_10hz": future,
            "future_valid_mask_10hz": state.future_valid,
            "ego_status": {
                "ego_velocity": state.velocity,
                "ego_acceleration": state.acceleration,
                "driving_command": state.command_one_hot(),
            },
            "nav_command": state.nav_command,
        },
    }


def driving_scene(state: EgoState, images: dict[str, list[Any]], token: str = ""):
    """qwen_drive.DrivingScene from {view tag: [4 images (PIL or RGB arrays), oldest first]}."""
    from PIL import Image
    from qwen_drive import CameraFrame, DrivingScene

    def frame(img):
        return CameraFrame(Image.fromarray(img) if isinstance(img, np.ndarray) else img)

    return DrivingScene(
        views={tag: [frame(i) for i in images[tag]] for tag in VIEW_TAGS},
        history=np.asarray(state.history, dtype=np.float64),
        history_velocity=np.asarray(state.history_velocity, dtype=np.float32),
        history_acceleration=np.asarray(state.history_acceleration, dtype=np.float32),
        ego_velocity=state.velocity,
        ego_acceleration=state.acceleration,
        driving_command=state.command_one_hot(),
        nav_command=state.nav_command,
        token=token,
    )


def read_predictions(path: str) -> dict[str, dict]:
    """Qwen-Drive run_planning.py output → {token: record}."""
    import json

    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue            # truncated last line from an interrupted run
            out[str(rec.get("token", ""))] = rec
    return out


def frame_from_token(token: str) -> Optional[int]:
    try:
        return int(token.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return None
