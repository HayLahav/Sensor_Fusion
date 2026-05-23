"""Tile annotated frame + BEV + text panel into a single output image."""
from __future__ import annotations
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import ReasoningOutput


def composite(
    annotated_frame: np.ndarray,
    bev_image: np.ndarray,
    reasoning: ReasoningOutput | None,
    target_width: int = 1280,
) -> np.ndarray:
    """
    Produce side-by-side composite: [annotated_frame | bev | text_panel].
    All panels resized to the same height.
    """
    h = annotated_frame.shape[0]

    # Resize BEV to match frame height
    bev_h = cv2.resize(bev_image, (h, h))

    # Build text panel
    panel_w = target_width - annotated_frame.shape[1] - h
    panel_w = max(panel_w, 200)
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

    if reasoning:
        lines = [
            "=== SCENE REASONING ===",
            "",
            *reasoning.summary.split(". "),
            "",
            "Anomalies:",
            *[f"  - {a}" for a in reasoning.anomalies],
            "",
            f"Latency: {reasoning.latency_ms:.0f}ms",
            f"Mode: {'visual' if reasoning.visual_mode else 'text'}",
        ]
        for i, line in enumerate(lines[:30]):
            y = 16 + i * 16
            if y > h - 8:
                break
            cv2.putText(panel, line, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    return np.concatenate([annotated_frame, bev_h, panel], axis=1)
