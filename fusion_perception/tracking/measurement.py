"""Adaptive measurement vector and covariance for KalmanCoWTracker."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from fusion_perception.utils.dataclasses import Detection3D


@dataclass
class MeasurementConfig:
    depth_alpha: float = 0.005    # σ_z = α·z² + β
    depth_beta: float = 0.5
    sigma_xy_base: float = 0.5
    sigma_xy_slope: float = 0.01
    cow_sigma_xy: float = 0.1     # CoWTracker lateral noise (metres)
    fallback_inflate: float = 3.0 # inflate R when CoW unavailable


def synthesize_measurement(
    det: Detection3D,
    cow_disp_px: Optional[np.ndarray],  # [dx, dy] median pixel displacement
    cow_valid: bool,
    K: np.ndarray,                       # 3×3 intrinsics
    cfg: MeasurementConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build measurement vector z (7-dim) and covariance R (7×7).
    box_3d layout: [cx, cy, cz, w, h, l, ry]
    KF state layout: [cx, cy, cz, θ, l, w, h]
    """
    b = det.box_3d   # [cx, cy, cz, w, h, l, ry]
    cx, cy, cz = b[0], b[1], b[2]
    cz_safe = max(float(cz), 0.1)  # guard against zero/negative monocular depth
    theta = b[6]
    l, w, h = b[5], b[3], b[4]

    z = np.array([cx, cy, cz, theta, l, w, h], dtype=np.float64)

    # Depth-dependent measurement noise
    sigma_z = cfg.depth_alpha * cz_safe ** 2 + cfg.depth_beta
    sigma_xy = cfg.sigma_xy_base + cfg.sigma_xy_slope * cz_safe
    R_base = np.diag([
        sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2,
        0.05, 0.09, 0.09, 0.09,
    ]).astype(np.float64)

    # Detection confidence scaling: lower score → larger R
    score = max(float(det.score), 0.1)
    R_det = R_base / score

    if cow_valid and cow_disp_px is not None:
        # Back-project pixel displacement to camera-frame 3D offset
        fx, fy = float(K[0, 0]), float(K[1, 1])
        dx_3d = float(cow_disp_px[0]) * cz_safe / fx
        dy_3d = float(cow_disp_px[1]) * cz_safe / fy
        z[0] += dx_3d
        z[1] += dy_3d

        # CoW measurement noise (visually accurate, but depth still uncertain)
        R_cow = np.diag([
            cfg.cow_sigma_xy ** 2, cfg.cow_sigma_xy ** 2, sigma_z ** 2,
            0.05, 0.09, 0.09, 0.09,
        ]).astype(np.float64)

        # Information-theoretic fusion: R_eff^-1 = R_det^-1 + R_cow^-1
        # NEW — element-wise on diagonals (10x faster, no LinAlgError risk)
        inv_R_det = 1.0 / np.diag(R_det)
        inv_R_cow = 1.0 / np.diag(R_cow)
        R_eff = np.diag(1.0 / (inv_R_det + inv_R_cow))
    else:
        R_eff = R_det * cfg.fallback_inflate

    return z, R_eff
