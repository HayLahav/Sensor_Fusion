"""Export KITTI-360 frames as Qwen-Drive planning scenes.

Writes, under --out:
  scenes.jsonl            one Qwen-Drive scene record per exported frame
  frames/<view>/<frame>.jpg   the camera views (front + virtual side views)

Then plan with Qwen-Drive in its own environment:
  cd Qwen-Drive-1.0 && PYTHONPATH=src python scripts/run_planning.py \\
      --model Qwen-Drive-1.0-4B --planner Qwen-Drive-1.0-4B/planner-rl \\
      --scenes <out>/scenes.jsonl --image-root <out> \\
      --output <out>/predictions.jsonl --mode reasoning_planning \\
      --attn-implementation sdpa
and check the plans in this pipeline with planning.qwen_drive.predictions
pointing at predictions.jsonl (see configs/default.yaml).

Example:
  python scripts/export_qwen_drive_scenes.py --kitti360-root /data/KITTI-360 \\
      --seq 2013_05_28_drive_0000_sync --start 0 --end 1000 --stride 10 \\
      --out outputs/qwen_drive/drive_0000
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_perception.data.kitti360_labels import _load_poses  # noqa: E402
from fusion_perception.data.kitti_calibration import Kitti360Calib  # noqa: E402
from fusion_perception.planning.ego_state import EgoStateExtractor  # noqa: E402
from fusion_perception.planning.kitti360_views import VIEW_TAGS, Kitti360ViewRenderer  # noqa: E402
from fusion_perception.planning.qwen_drive_scene import (  # noqa: E402
    camera_frame_offsets, scene_record,
)
from fusion_perception.tracking.ego_motion import PoseEgoCompensation  # noqa: E402

_VIEW_DIR = {VIEW_TAGS[0]: "front", VIEW_TAGS[1]: "front_left", VIEW_TAGS[2]: "front_right"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kitti360-root", required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="last frame (inclusive); default: last pose")
    p.add_argument("--stride", type=int, default=10, help="export every Nth frame")
    p.add_argument("--side-mode", default="fisheye", choices=["fisheye", "front_crop", "blank"])
    p.add_argument("--side-yaw", type=float, default=55.0, help="virtual side camera yaw (deg)")
    p.add_argument("--side-hfov", type=float, default=75.0, help="virtual side camera field of view (deg)")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = Kitti360Calib.from_dir(os.path.join(a.kitti360_root, "calibration"))
    poses = _load_poses(os.path.join(a.kitti360_root, "data_poses", a.seq, "poses.txt"))
    ego = PoseEgoCompensation(poses, calib.T_pose_to_cam, calib.R_rect)
    extractor = EgoStateExtractor(ego, fps=a.fps)
    renderer = Kitti360ViewRenderer(a.kitti360_root, a.seq, side_mode=a.side_mode,
                                    side_yaw_deg=a.side_yaw, side_hfov_deg=a.side_hfov)
    offsets = camera_frame_offsets(a.fps)
    end = a.end if a.end is not None else max(poses)

    def frame_path(tag: str, f: int) -> str:
        return f"frames/{_VIEW_DIR[tag]}/{f:010d}.jpg"

    def ensure_frame(f: int) -> bool:
        if all((out / frame_path(t, f)).exists() for t in VIEW_TAGS):
            return True
        views = renderer.render(f)
        if views is None:
            return False
        for t in VIEW_TAGS:
            p = out / frame_path(t, f)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), cv2.cvtColor(views[t], cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
        return True

    n_written = n_skipped = 0
    commands: dict[str, int] = {}
    with open(out / "scenes.jsonl", "w") as fh:
        for f in range(a.start, end + 1, a.stride):
            state = extractor.get(f)
            if state is None or not all(ensure_frame(f + k) for k in offsets):
                n_skipped += 1
                continue
            paths = {t: [frame_path(t, f + k) for k in offsets] for t in VIEW_TAGS}
            fh.write(json.dumps(scene_record(a.seq, state, paths)) + "\n")
            commands[state.command] = commands.get(state.command, 0) + 1
            n_written += 1

    print(f"Wrote {n_written} scenes to {out / 'scenes.jsonl'} "
          f"({n_skipped} skipped: missing poses or images). Commands: {commands}")
    print("\nNext, in the Qwen-Drive environment:\n"
          f"  PYTHONPATH=src python scripts/run_planning.py --model Qwen-Drive-1.0-4B \\\n"
          f"      --planner Qwen-Drive-1.0-4B/planner-rl --scenes {out.resolve()}/scenes.jsonl \\\n"
          f"      --image-root {out.resolve()} --output {out.resolve()}/predictions.jsonl \\\n"
          f"      --mode reasoning_planning --attn-implementation sdpa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
