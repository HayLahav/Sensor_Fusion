"""Draw instance segmentation contours, car keypoints, track IDs, and trajectory tails.

3D wireframe boxes are intentionally omitted — they were inaccurate due to noisy
LiDAR frustum fitting.  Instead each detected vehicle shows:
  - Instance segmentation boundary (YOLO26n-seg mask contour, unique color per instance)
  - ApolloCar3D keypoints: green = front group, red = rear group
  - Track ID + class + motion label near the centroid
"""
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D, Track, OccupancyGrid
from fusion_perception.utils.geometry import world_to_grid

_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

# ApolloCar3D 24-keypoint groups
_KP_FRONT = [0, 1, 2, 3, 4, 5, 7, 19]
_KP_REAR  = [8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]
_KP_CONF  = 0.15


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _motion_str(track: Track, motion_val: float = 0.0) -> str:
    """Return motion label. motion_val is the BEV motion-grid value at track's cell."""
    if motion_val < 0.3:
        return "stationary"
    vx, _, vz = track.velocity_3d
    speed = (vx ** 2 + vz ** 2) ** 0.5
    if speed < 0.5:
        return "moving"
    if vz < -1.0:
        return "approaching"
    if vz > 1.0:
        return "receding"
    return "moving"


def _draw_instance_masks(
    out: np.ndarray,
    instance_masks: list[tuple[np.ndarray, int]],
) -> None:
    """Draw per-instance segmentation contours on the frame."""
    for i, (mask, _cls_id) in enumerate(instance_masks):
        color = _PALETTE[i % len(_PALETTE)]
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Semi-transparent fill
        overlay = out.copy()
        cv2.fillPoly(overlay, contours, color)
        cv2.addWeighted(overlay, 0.20, out, 0.80, 0, out)
        # Solid contour
        cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)


def _draw_keypoints(out: np.ndarray, kp_annotations: list) -> None:
    """Draw front (green) and rear (red) ApolloCar3D keypoints."""
    H, W = out.shape[:2]
    for ann in kp_annotations:
        try:
            kps = np.array(ann.data, dtype=np.float32)
        except Exception:
            continue
        if kps.shape[0] < 24:
            continue
        for idx in _KP_FRONT:
            x, y, c = kps[idx]
            if c >= _KP_CONF:
                cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1, cv2.LINE_AA)
        for idx in _KP_REAR:
            x, y, c = kps[idx]
            if c >= _KP_CONF:
                cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)


def _get_motion_val(track: Track, occupancy: OccupancyGrid | None) -> float:
    """Look up the BEV motion-grid value at a track's position."""
    if occupancy is None or occupancy.motion_grid is None or not track.position_3d_history:
        return 0.0
    x, _, z = track.position_3d_history[-1]
    rc = world_to_grid(x, z, occupancy.x_range, occupancy.z_range, occupancy.resolution)
    if rc is None:
        return 0.0
    try:
        return float(occupancy.motion_grid[rc[0]][rc[1]])
    except (IndexError, TypeError):
        return 0.0


def annotate_frame(
    frame: np.ndarray,
    tracks: list[Track],
    detections: list[Detection3D],
    reasoning_text: str = "",
    intrinsics: Optional[np.ndarray] = None,
    kp_annotations: list | None = None,
    instance_masks: list[tuple[np.ndarray, int]] | None = None,
    occupancy: OccupancyGrid | None = None,
) -> np.ndarray:
    """Return annotated copy of frame.

    Draws (in order, bottom to top):
      1. Instance segmentation contours + semi-transparent fills
      2. ApolloCar3D keypoints (front=green, rear=red)
      3. Trajectory tails for confirmed tracks
      4. Track ID + class + motion label
      5. Raw detection count
    """
    out = frame.copy()

    # 1. Instance segmentation contours
    if instance_masks:
        _draw_instance_masks(out, instance_masks)

    # 2. Car keypoints
    if kp_annotations:
        _draw_keypoints(out, kp_annotations)

    # 3. Trajectory tails — only for objects confirmed moving by motion_grid.
    # Pixel-space centroids drift for parked cars as ego drives past them,
    # creating misleading lines. Suppress tails for stationary objects.
    for track in tracks:
        if _get_motion_val(track, occupancy) < 0.3:
            continue
        color = _color(track.track_id)
        hist = track.centroid_history[-20:]
        for i in range(1, len(hist)):
            p1 = (int(hist[i-1][0]), int(hist[i-1][1]))
            p2 = (int(hist[i][0]),   int(hist[i][1]))
            alpha = i / len(hist)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(out, p1, p2, faded, 2)

    # 4. Track label (ID + class + motion) near centroid
    for track in tracks:
        color  = _color(track.track_id)
        mv     = _get_motion_val(track, occupancy)
        motion = _motion_str(track, mv)
        label  = f"#{track.track_id} {track.class_name} {motion}"
        if track.centroid_history:
            cx_px = int(track.centroid_history[-1][0])
            cy_px = max(int(track.centroid_history[-1][1]) - 8, 12)
        elif track.last_box_2d is not None:
            x1, y1, x2, y2 = track.last_box_2d
            cx_px = int((x1 + x2) / 2)
            cy_px = max(int(y1) - 6, 12)
        else:
            continue
        cv2.putText(out, label, (cx_px, cy_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # 5. Detection count
    if detections:
        cv2.putText(out, f"det:{len(detections)}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    return out
