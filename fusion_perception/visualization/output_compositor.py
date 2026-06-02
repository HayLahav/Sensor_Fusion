"""Tile annotated frame + BEV + (depth+keypoints) + (semantic+instances) + text panel."""
from __future__ import annotations
import cv2
import numpy as np
from fusion_perception.utils.dataclasses import ReasoningOutput

_DEPTH_MAX_M = 80.0

# Cityscapes trainId → BGR (shared with bev_renderer)
_SEM_BGR: dict[int, tuple[int, int, int]] = {
    -1: ( 40,  40,  40),
     0: (128,  64, 128),   # road
     1: (232,  35, 244),   # sidewalk
     2: ( 70,  70,  70),   # building
     8: ( 35, 142, 107),   # vegetation
     9: (152, 251, 152),   # terrain
    11: ( 60,  20, 220),   # person
    12: (  0,   0, 255),   # rider
    13: (142,   0,   0),   # car
    14: ( 70,   0,   0),   # truck
    15: (100,  60,   0),   # bus
    17: (230,   0,   0),   # motorcycle
    18: ( 32,  11, 119),   # bicycle
}

# Bright per-instance palette (BGR) for instance segmentation contours
_INST_PALETTE: list[tuple[int, int, int]] = [
    (  0, 255,   0), (  0, 200, 255), (255, 128,   0), (255,   0, 255),
    (  0, 255, 200), (255, 255,   0), (128,   0, 255), (  0, 128, 255),
    (255,  64, 128), ( 64, 255, 128), (255, 200,   0), (200,   0, 255),
]

# ApolloCar3D keypoint indices for front / rear groups
_KP_FRONT = [0, 1, 2, 3, 4, 5, 7, 19]   # front_up, front_light, front_low, front_wheel
_KP_REAR  = [8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]
_KP_CONF_MIN = 0.15


# ── Depth + keypoints panel ───────────────────────────────────────────────────

def _colorize_depth(depth_map: np.ndarray, target_h: int) -> np.ndarray:
    """[H,W] float32 → square BGR PLASMA panel (no keypoints)."""
    d = np.clip(depth_map, 0.0, _DEPTH_MAX_M)
    d_u8 = (d / _DEPTH_MAX_M * 255.0).astype(np.uint8)
    panel = cv2.resize(
        cv2.applyColorMap(d_u8, cv2.COLORMAP_PLASMA),
        (target_h, target_h), interpolation=cv2.INTER_AREA,
    )
    return panel


def _draw_keypoints_on_depth(
    panel: np.ndarray,
    kp_annotations: list,
    frame_hw: tuple[int, int],
) -> np.ndarray:
    """Overlay car keypoints and heading arrows on an already-colorized depth panel."""
    if not kp_annotations:
        return panel

    fH, fW = frame_hw
    pH, pW = panel.shape[:2]
    sx = pW / fW
    sy = pH / fH

    for ann in kp_annotations:
        try:
            kps = np.array(ann.data, dtype=np.float32)  # [24, 3]
        except Exception:
            continue
        if kps.shape[0] < 24:
            continue

        front_pts, rear_pts = [], []

        for idx in _KP_FRONT:
            x, y, c = kps[idx]
            if c < _KP_CONF_MIN:
                continue
            px, py = int(x * sx), int(y * sy)
            cv2.circle(panel, (px, py), 3, (0, 255, 0), -1)   # green = front
            front_pts.append((px, py))

        for idx in _KP_REAR:
            x, y, c = kps[idx]
            if c < _KP_CONF_MIN:
                continue
            px, py = int(x * sx), int(y * sy)
            cv2.circle(panel, (px, py), 3, (0, 0, 255), -1)   # red = rear
            rear_pts.append((px, py))

        # Draw heading arrow: rear centroid → front centroid
        if len(front_pts) >= 2 and len(rear_pts) >= 2:
            fc = (int(np.mean([p[0] for p in front_pts])),
                  int(np.mean([p[1] for p in front_pts])))
            rc = (int(np.mean([p[0] for p in rear_pts])),
                  int(np.mean([p[1] for p in rear_pts])))
            cv2.arrowedLine(panel, rc, fc, (0, 255, 255), 2, tipLength=0.25)

    return panel


def build_depth_panel(
    depth_map: np.ndarray,
    target_h: int,
    kp_annotations: list | None = None,
    frame_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """PLASMA depth + optional ApolloCar3D keypoint overlay."""
    panel = _colorize_depth(depth_map, target_h)
    cv2.putText(panel, "DEPTH+KP" if kp_annotations else "DEPTH", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"0-{int(_DEPTH_MAX_M)}m", (6, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    if kp_annotations and frame_hw is not None:
        panel = _draw_keypoints_on_depth(panel, kp_annotations, frame_hw)
    return panel


# ── Semantic + instance segmentation panel ────────────────────────────────────

def build_sem_panel(
    sem_mask: np.ndarray | None,
    target_h: int,
    instance_masks: list[tuple[np.ndarray, int]] | None = None,
) -> np.ndarray:
    """Cityscapes semantic colours + optional instance segmentation contours.

    instance_masks: list of (bool [H,W] mask, COCO cls_id) per detection.
    Contours are drawn in unique bright colours so individual instances stand
    out even when they share the same semantic class (e.g. two adjacent cars).
    """
    # ── Base semantic layer ───────────────────────────────────────────────────
    if sem_mask is None:
        panel = np.full((target_h, target_h, 3), 20, dtype=np.uint8)
        cv2.putText(panel, "SEMANTIC", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        return panel

    h_src, w_src = sem_mask.shape[:2]
    colored = np.full((h_src, w_src, 3), _SEM_BGR[-1], dtype=np.uint8)
    for cls_id, bgr in _SEM_BGR.items():
        mask = sem_mask == cls_id
        if mask.any():
            colored[mask] = bgr

    panel = cv2.resize(colored, (target_h, target_h), interpolation=cv2.INTER_NEAREST)

    # ── Instance segmentation overlay ────────────────────────────────────────
    if instance_masks:
        for inst_i, (inst_mask, _cls_id) in enumerate(instance_masks):
            color = _INST_PALETTE[inst_i % len(_INST_PALETTE)]

            # Scale mask to panel size
            mask_small = cv2.resize(
                inst_mask.astype(np.uint8),
                (target_h, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Semi-transparent fill
            fill = np.zeros_like(panel)
            fill[mask_small > 0] = color
            cv2.addWeighted(panel, 1.0, fill, 0.25, 0, panel)

            # Solid contour outline
            contours, _ = cv2.findContours(
                mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(panel, contours, -1, color, 2)

    label = "SEM+INST" if instance_masks else "SEMANTIC"
    cv2.putText(panel, label, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


# ── Subtitle caption bar ─────────────────────────────────────────────────────

def _draw_caption_bar(
    composite: np.ndarray,
    reasoning: ReasoningOutput,
) -> np.ndarray:
    """Burn a semi-transparent caption bar into the bottom of the composite.

    Line 1 (white)  : SUMMARY
    Line 2 (cyan)   : TRAJECTORY
    Line 3 (orange) : CAUTION  — only when an anomaly is present
    """
    H, W = composite.shape[:2]
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if reasoning.trajectory_nl:
        lines.append((f"TRAJECTORY: {reasoning.trajectory_nl}", (255, 230, 0)))
    if reasoning.anomalies:
        lines.append((f"CAUTION: {reasoning.anomalies[0]}", (0, 140, 255)))

    if not lines:
        return composite

    line_h   = 22
    padding  = 8
    bar_h    = len(lines) * line_h + padding * 2
    bar_y    = H - bar_h

    # Semi-transparent dark background
    overlay = composite.copy()
    cv2.rectangle(overlay, (0, bar_y), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, composite, 0.38, 0, composite)

    for i, (text, color) in enumerate(lines):
        y = bar_y + padding + (i + 1) * line_h - 4
        cv2.putText(composite, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    return composite


# ── Compositor ────────────────────────────────────────────────────────────────

def composite(
    annotated_frame: np.ndarray,
    bev_image: np.ndarray,
    reasoning: ReasoningOutput | None,
    depth_map: np.ndarray | None = None,
    sem_mask: np.ndarray | None = None,
    show_sem: bool = False,
    target_width: int = 1280,
    kp_annotations: list | None = None,
    instance_masks: list[tuple[np.ndarray, int]] | None = None,
) -> np.ndarray:
    """
    Layout: [annotated_frame | BEV | depth+kp | semantic+inst | text_panel]

    depth+kp    : shown when depth_map is not None; keypoints overlaid when available.
    semantic+inst: shown when show_sem=True; instance contours overlaid when available.
    """
    h = annotated_frame.shape[0]
    frame_hw = annotated_frame.shape[:2]

    bev_h = cv2.resize(bev_image, (h, h))

    depth_panel = (
        build_depth_panel(depth_map, h, kp_annotations, frame_hw)
        if depth_map is not None else None
    )
    sem_panel = (
        build_sem_panel(sem_mask, h, instance_masks)
        if show_sem else None
    )

    extra_w = h * (1 if depth_panel is not None else 0) + h * (1 if sem_panel is not None else 0)
    panel_w = max(target_width - annotated_frame.shape[1] - h - extra_w, 200)
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

    if reasoning:
        lines = [
            "=== SCENE REASONING ===", "",
            *reasoning.summary.split(". "), "",
            *([f"CAUTION: {reasoning.anomalies[0]}"] if reasoning.anomalies else []),
            "",
            reasoning.trajectory_nl if reasoning.trajectory_nl else "",
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

    panels = [annotated_frame, bev_h]
    if depth_panel is not None:
        panels.append(depth_panel)
    if sem_panel is not None:
        panels.append(sem_panel)
    panels.append(panel)
    out = np.concatenate(panels, axis=1)

    if reasoning:
        out = _draw_caption_bar(out, reasoning)

    return out
