"""LiDAR-anchored depth correction for per-detection depth refinement.

Two complementary corrections:

  1. Global scale:
       Project all LiDAR points into the image and sample the monocular depth
       map at those pixels.  Compute  scale = median(z_lidar / z_mono).
       Apply scale to the entire depth map.  This corrects systematic scale
       drift in the monocular model.

  2. Per-detection frustum anchoring:
       For each detection box, collect the LiDAR points whose projections fall
       inside the 2D bounding box.  If enough points are present, replace the
       detection depth with median(z_lidar_in_frustum).  Also update
       centroid_3d and the box_3d centroid to match.

Together they give:
  - Dense, high-resolution depth from DA-V2 (good for BEV background)
  - Accurate metric depth per tracked object from LiDAR (good for KF init)
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("depth.lidar_anchor")


def _project_lidar_to_image(
    lidar_pts_cam: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame LiDAR points to pixel coords.

    Returns (uv, mask) where uv is [M, 2] float32 (u=col, v=row)
    and mask indexes into lidar_pts_cam for forward-facing points only.
    """
    z_mask = lidar_pts_cam[:, 2] > 0.1
    pts = lidar_pts_cam[z_mask]
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32), z_mask

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * pts[:, 0] / pts[:, 2] + cx
    v = fy * pts[:, 1] / pts[:, 2] + cy
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv, z_mask


class LidarDepthAnchor:
    """Corrects monocular depth estimates using LiDAR measurements.

    Parameters
    ----------
    min_frustum_points : int
        Minimum LiDAR points inside a 2D box to use frustum anchoring.
    global_scale : bool
        Whether to compute and apply a global scale correction first.
    scale_clamp : tuple[float, float]
        Allowed range for the global scale factor.
    """

    def __init__(
        self,
        min_frustum_points: int = 5,
        global_scale: bool = True,
        scale_clamp: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        self.min_frustum_points = min_frustum_points
        self.global_scale = global_scale
        self.scale_clamp = scale_clamp

    def compute_scale(
        self,
        depth_map: np.ndarray,
        lidar_pts_cam: np.ndarray,
        intrinsics: np.ndarray,
    ) -> float:
        """Compute global scale = median(z_lidar / z_mono) over visible LiDAR points.

        Returns 1.0 if no valid correspondences are found.
        """
        h, w = depth_map.shape
        uv, z_mask = _project_lidar_to_image(lidar_pts_cam, intrinsics)
        pts = lidar_pts_cam[z_mask]

        if len(pts) == 0:
            return 1.0

        # Keep points inside the image
        img_mask = (
            (uv[:, 0] >= 0) & (uv[:, 0] < w) &
            (uv[:, 1] >= 0) & (uv[:, 1] < h)
        )
        uv_in = uv[img_mask].astype(np.int32)
        z_lidar = pts[img_mask, 2]
        if len(z_lidar) == 0:
            return 1.0

        z_mono = depth_map[uv_in[:, 1], uv_in[:, 0]]
        valid = (z_mono > 0.1) & (z_lidar > 0.1)
        if valid.sum() < 10:
            return 1.0

        ratios = z_lidar[valid] / z_mono[valid]
        scale = float(np.median(ratios))
        scale = float(np.clip(scale, self.scale_clamp[0], self.scale_clamp[1]))
        logger.debug(f"LiDAR scale correction: {scale:.3f} ({valid.sum()} correspondences)")
        return scale

    def refine_depth_map(
        self,
        depth_map: np.ndarray,
        lidar_pts_cam: np.ndarray,
        intrinsics: np.ndarray,
    ) -> np.ndarray:
        """Apply global scale correction to the depth map in-place (returns new array)."""
        if not self.global_scale:
            return depth_map
        scale = self.compute_scale(depth_map, lidar_pts_cam, intrinsics)
        return (depth_map * scale).astype(np.float32)

    def anchor_detections(
        self,
        detections: list[Detection3D],
        depth_map: Optional[np.ndarray],
        lidar_pts_cam: Optional[np.ndarray],
        intrinsics: np.ndarray,
    ) -> list[Detection3D]:
        """Refine depth and centroid_3d for each detection.

        Priority per detection:
          1. Median LiDAR depth in 2D frustum (if ≥ min_frustum_points)
          2. Median monocular depth in 2D box (if depth_map available)
          3. Keep original WildDet3D depth (fallback)
        """
        if not detections:
            return detections

        h_img, w_img = (depth_map.shape if depth_map is not None else (0, 0))

        # Pre-project LiDAR once for all detections
        lidar_uv: Optional[np.ndarray] = None
        lidar_z: Optional[np.ndarray] = None
        if lidar_pts_cam is not None and len(lidar_pts_cam) > 0:
            uv, z_mask = _project_lidar_to_image(lidar_pts_cam, intrinsics)
            pts_fwd = lidar_pts_cam[z_mask]
            if len(pts_fwd) > 0:
                lidar_uv = uv          # [M, 2] (u, v)
                lidar_z = pts_fwd[:, 2]  # [M]

        refined: list[Detection3D] = []
        for det in detections:
            x1, y1, x2, y2 = det.box_2d
            new_depth: Optional[float] = None

            # ── LiDAR frustum ────────────────────────────────────────────────
            if lidar_uv is not None and lidar_z is not None:
                frust_mask = (
                    (lidar_uv[:, 0] >= x1) & (lidar_uv[:, 0] <= x2) &
                    (lidar_uv[:, 1] >= y1) & (lidar_uv[:, 1] <= y2)
                )
                frust_z = lidar_z[frust_mask]
                if len(frust_z) >= self.min_frustum_points:
                    new_depth = float(np.median(frust_z))
                    logger.debug(
                        f"Det {det.class_name}: LiDAR frustum depth "
                        f"{new_depth:.1f}m ({len(frust_z)} pts)"
                    )

            # ── Monocular depth fallback ──────────────────────────────────────
            if new_depth is None and depth_map is not None and h_img > 0:
                ix1 = max(0, int(x1))
                iy1 = max(0, int(y1))
                ix2 = min(w_img - 1, int(x2))
                iy2 = min(h_img - 1, int(y2))
                if ix2 > ix1 and iy2 > iy1:
                    patch = depth_map[iy1:iy2, ix1:ix2]
                    valid = patch[patch > 0.1]
                    if len(valid) > 0:
                        new_depth = float(np.median(valid))
                        logger.debug(
                            f"Det {det.class_name}: mono depth "
                            f"{new_depth:.1f}m"
                        )

            if new_depth is None or new_depth < 0.5:
                refined.append(det)
                continue

            # Recompute centroid_3d using new depth and camera intrinsics.
            # Original centroid_2d gives the pixel; backproject with new z.
            cx_px, cy_px = float(det.centroid_2d[0]), float(det.centroid_2d[1])
            fx = float(intrinsics[0, 0])
            fy = float(intrinsics[1, 1])
            cx_k = float(intrinsics[0, 2])
            cy_k = float(intrinsics[1, 2])
            x3 = (cx_px - cx_k) * new_depth / fx
            y3 = (cy_px - cy_k) * new_depth / fy
            new_c3d = [x3, y3, new_depth]

            # Update box_3d centroid (keep w, h, l, ry from WildDet3D)
            old_b3d = list(det.box_3d)
            new_b3d = [x3, y3, new_depth] + old_b3d[3:]

            import copy
            det2 = copy.copy(det)
            det2.depth = new_depth
            det2.centroid_3d = new_c3d
            det2.box_3d = new_b3d
            refined.append(det2)

        return refined
