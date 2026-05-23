"""Draw 2D boxes, track IDs, scores, and trajectory tails on video frames."""
from __future__ import annotations
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D, Track

_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def _color(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def annotate_frame(
    frame: np.ndarray,
    tracks: list[Track],
    detections: list[Detection3D],
    reasoning_text: str = "",
) -> np.ndarray:
    """Return annotated copy of frame with boxes, IDs, and trajectories."""
    out = frame.copy()

    # Draw trajectory tails
    for track in tracks:
        color = _color(track.track_id)
        hist = track.centroid_history[-20:]
        for i in range(1, len(hist)):
            p1 = (int(hist[i-1][0]), int(hist[i-1][1]))
            p2 = (int(hist[i][0]), int(hist[i][1]))
            alpha = i / len(hist)
            faded = tuple(int(c * alpha) for c in color)
            cv2.line(out, p1, p2, faded, 2)

    # Draw detection boxes
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box_2d]
        tid = next((t.track_id for t in tracks if t.class_name == det.class_name
                    and abs(t.cow_query_point[0] - det.centroid_2d[0]) < 30), -1)
        color = _color(tid) if tid >= 0 else (200, 200, 200)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {det.class_name} {det.score:.2f} d={det.depth:.1f}m"
        cv2.putText(out, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Overlay reasoning text
    if reasoning_text:
        for i, line in enumerate(reasoning_text.split(". ")[:3]):
            cv2.putText(out, line.strip(), (8, out.shape[0] - 12 - i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return out
