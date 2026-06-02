"""Frustum-based 3D detector: YOLO 2D → multi-modal fusion → 3D bounding box.

Data streams used (in priority order per detection)
────────────────────────────────────────────────────
1. LiDAR + semantic mask  (primary)
   Project LiDAR into the image; keep only points whose pixel lands on the
   same Cityscapes class as the YOLO detection (car→13, person→11/12, etc.).
   Fit a 3D box to those class-consistent points.

2. Depth map + semantic mask  (fallback when LiDAR is sparse)
   Take every pixel inside the 2D box that matches the semantic class, read
   its depth value, and back-project to camera-frame 3D with the pinhole
   model.  The semantic mask gives a tight object boundary; the depth map
   gives metric Z.  Fit a 3D box to the back-projected point cloud.

3. LiDAR frustum only  (fallback when sem_mask is None)
   Use all LiDAR points inside the 2D frustum, cleaned with depth clustering
   and a ground-removal heuristic.

4. Class-typical dimensions  (last resort)
   When the frustum is too sparse for fitting, use the median LiDAR depth
   plus canonical w/h/l for the detected class.

No compilation required — depends only on ultralytics (already installed).
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from fusion_perception.models.base_detector import BaseDetector
from fusion_perception.utils.dataclasses import Detection3D
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("frustum_detector")

# ── Class mappings ────────────────────────────────────────────────────────────

_COCO_TO_CLASS: dict[int, str] = {
    0: "person",
    1: "cyclist",       # COCO: bicycle
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

_CLASS_TO_ID: dict[str, int] = {
    "car": 0, "person": 1, "cyclist": 2,
    "truck": 3, "bus": 4, "motorcycle": 5,
}

# Typical [w, h, l] metres (lateral, vertical, forward)
_CLASS_DIMS: dict[str, tuple[float, float, float]] = {
    "car":        (1.8, 1.5, 4.5),
    "person":     (0.6, 1.7, 0.6),
    "cyclist":    (0.8, 1.7, 1.8),
    "truck":      (2.5, 2.5, 8.0),
    "bus":        (2.5, 3.0, 10.0),
    "motorcycle": (0.8, 1.5, 2.0),
}

# Cityscapes trainId classes per detection category used for semantic filtering
_CLASS_SEM_IDS: dict[str, tuple[int, ...]] = {
    "car":        (13,),
    "person":     (11, 12),    # person + rider
    "cyclist":    (18, 12),    # bicycle + rider
    "truck":      (14,),
    "bus":        (15,),
    "motorcycle": (17,),
}

_CAM_HEIGHT_M = 1.5   # approximate camera mounting height above road (metres)
_MIN_DEPTH_PX = 10    # minimum back-projected pixels to attempt depth-based fit


# ── Box fitting ───────────────────────────────────────────────────────────────

def _fit_box(
    pts: np.ndarray,
    class_name: str = "car",
    skip_ground_filter: bool = False,
) -> tuple[float, float, float, float, float, float, float]:
    """Fit [cx, cy, cz, w, h, l, ry] from camera-frame points [N, 3].

    Cleaning stages:
      1. Depth clustering — keep [0.6, 1.5] × median_z (removes ground/BG).
         Skipped for depth-map back-projected points (already class-filtered).
      2. Ground removal — drop Y_cam ≥ _CAM_HEIGHT_M (road surface heuristic).
      3. Dimension cap — clamp to 1.5 × class-typical size.
    """
    if len(pts) < 3:
        typ_w, typ_h, typ_l = _CLASS_DIMS.get(class_name, (1.8, 1.5, 4.5))
        cx, cy = float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
        cz = float(np.median(pts[:, 2]))
        return cx, cy, cz, typ_w, typ_h, typ_l, 0.0

    # 1. Depth clustering (heuristic; skip for pre-filtered depth-map clouds)
    if not skip_ground_filter:
        z_med = float(np.median(pts[:, 2]))
        z_keep = (pts[:, 2] >= z_med * 0.6) & (pts[:, 2] <= z_med * 1.5)
        if z_keep.sum() >= 3:
            pts = pts[z_keep]

    # 2. Ground removal
    y_keep = pts[:, 1] < _CAM_HEIGHT_M
    if y_keep.sum() >= 3:
        pts = pts[y_keep]

    cx = float(np.median(pts[:, 0]))
    cy = float(np.median(pts[:, 1]))
    cz = float(np.median(pts[:, 2]))

    w_raw = max(0.3, float(np.percentile(pts[:, 0], 95) - np.percentile(pts[:, 0], 5)))
    h_raw = max(0.3, float(np.percentile(pts[:, 1], 95) - np.percentile(pts[:, 1], 5)))
    l_raw = max(0.3, float(np.percentile(pts[:, 2], 95) - np.percentile(pts[:, 2], 5)))

    # 3. Dimension cap
    typ_w, typ_h, typ_l = _CLASS_DIMS.get(class_name, (1.8, 1.5, 4.5))
    w = min(w_raw, typ_w * 1.5)
    h = min(h_raw, typ_h * 1.5)
    l = min(l_raw, typ_l * 1.5)

    # PCA heading in BEV x-z plane
    ry = 0.0
    if len(pts) >= 8:
        xz = pts[:, [0, 2]] - np.array([cx, cz])
        cov = np.cov(xz.T)
        if np.isfinite(cov).all() and cov.shape == (2, 2):
            _, vecs = np.linalg.eigh(cov)
            dominant = vecs[:, 1]
            ry = float(np.arctan2(dominant[0], dominant[1]))

    return cx, cy, cz, w, h, l, ry


# ── Detector ──────────────────────────────────────────────────────────────────

class FrustumLidarDetector(BaseDetector):
    """Multi-modal 3D detector: YOLO 2D + LiDAR + semantic + depth → Detection3D.

    Parameters
    ----------
    model_id : str
        Ultralytics YOLO model for 2D detection (used when seg_model_id is None).
    seg_model_id : str | None
        Ultralytics YOLO instance-segmentation model (e.g. ``yolo11n-seg.pt``).
        When set this model is used for both detection AND per-instance masks,
        replacing the rectangular box frustum with a tight pixel mask so that
        clustered vehicles (e.g. parked cars) get separate LiDAR point clouds.
    score_threshold : float
        Minimum YOLO 2D confidence to keep a detection.
    min_frustum_pts : int
        Minimum class-consistent LiDAR points needed to attempt direct fitting.
    min_depth_px : int
        Minimum semantic-masked depth-map pixels needed for depth-based fitting.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model_id: str = "yolo26n.pt",
        seg_model_id: str | None = "yolo26n-seg.pt",
        score_threshold: float = 0.30,
        min_frustum_pts: int = 5,
        min_depth_px: int = _MIN_DEPTH_PX,
        device: str = "cuda",
        use_keypoints: bool = True,
    ) -> None:
        self.model_id = model_id
        self.seg_model_id = seg_model_id
        self.score_threshold = score_threshold
        self.min_frustum_pts = min_frustum_pts
        self.min_depth_px = min_depth_px
        self.device = device
        self.use_keypoints = use_keypoints
        self._model = None
        self._seg_model = None
        self._kp_estimator = None
        # Cached per-frame outputs for visualization
        self.last_kp_annotations: list = []
        self.last_instance_masks: list[tuple[np.ndarray, str]] = []

    def load(self, checkpoint_path: str = "", device: str = "") -> None:
        from ultralytics import YOLO
        self._model = YOLO(self.model_id)
        if self.seg_model_id:
            self._seg_model = YOLO(self.seg_model_id)
        if self.use_keypoints:
            from fusion_perception.models.car_keypoint import CarKeypointEstimator
            self._kp_estimator = CarKeypointEstimator(device=device or self.device)
            self._kp_estimator.load()
        kp_status = "keypoints:ON" if (self._kp_estimator and self._kp_estimator.available) else "keypoints:OFF"
        msg = (
            f"FrustumLidarDetector: {self.model_id}"
            + (f" + seg:{self.seg_model_id}" if self.seg_model_id else "")
            + f" | {kp_status}"
            + f" | device={device or self.device}"
            + f" | score_thr={self.score_threshold} min_pts={self.min_frustum_pts}"
        )
        logger.info(msg)
        print(msg)

    def detect(
        self,
        frame: np.ndarray,
        frame_idx: int,
        intrinsics: Optional[np.ndarray],
        prompts: list[str],
        lidar_pts_velo: Optional[np.ndarray] = None,
        calib: object = None,
        sem_mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> list[Detection3D]:
        if self._model is None:
            raise RuntimeError("Call load() before detect()")

        H, W = frame.shape[:2]
        fx  = float(intrinsics[0, 0]) if intrinsics is not None else 552.0
        fy  = float(intrinsics[1, 1]) if intrinsics is not None else 552.0
        ppx = float(intrinsics[0, 2]) if intrinsics is not None else W / 2.0
        ppy = float(intrinsics[1, 2]) if intrinsics is not None else H / 2.0

        # ── Stream 1: YOLO 2D detection (+ optional instance segmentation) ─────
        # Use the seg model when available — it returns identical boxes but also
        # per-instance pixel masks that let us isolate individual vehicles even
        # when their bounding boxes overlap (e.g. parked cars side-by-side).
        _active_model = self._seg_model if self._seg_model is not None else self._model
        results = _active_model.predict(
            frame, conf=self.score_threshold, verbose=False, device=self.device,
        )
        boxes_2d = results[0].boxes
        if boxes_2d is None or len(boxes_2d) == 0:
            return []

        # Pre-compute per-instance binary masks at full image resolution.
        # mask_imgs[i] is a bool [H, W] array or None when unavailable.
        mask_imgs: list[np.ndarray | None] = [None] * len(boxes_2d)
        raw_masks = results[0].masks
        if raw_masks is not None and len(raw_masks) > 0:
            import torch, torch.nn.functional as F
            masks_t = raw_masks.data          # [N, Hm, Wm] float32 on GPU/CPU
            if masks_t.ndim == 3 and masks_t.shape[0] > 0:
                masks_full = F.interpolate(
                    masks_t.unsqueeze(0).float(),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )[0].cpu().numpy()            # [N, H, W]
                for i in range(min(len(boxes_2d), masks_full.shape[0])):
                    mask_imgs[i] = masks_full[i] > 0.5

        # ── Stream 2: project LiDAR into image ───────────────────────────────
        pts_cam: Optional[np.ndarray] = None
        us: Optional[np.ndarray] = None
        vs: Optional[np.ndarray] = None

        if lidar_pts_velo is not None and calib is not None and intrinsics is not None:
            pts_cam = calib.velo_to_cam(lidar_pts_velo[:, :3])   # [M, 3] z>0.1
            if len(pts_cam) > 0:
                K   = intrinsics.astype(np.float32)
                uvz = (K @ pts_cam.T).T
                us  = uvz[:, 0] / uvz[:, 2]
                vs  = uvz[:, 1] / uvz[:, 2]

        # ── Stream 3: car keypoints (one inference pass for the whole frame) ───
        # Annotations are matched per-box below to override PCA yaw with a
        # semantically grounded front→rear direction.
        kp_annotations: list = []
        if self._kp_estimator and self._kp_estimator.available:
            kp_annotations = self._kp_estimator.predict_frame(frame)

        # ── Per-box 3D lifting ────────────────────────────────────────────────
        detections: list[Detection3D] = []

        for det_i, box in enumerate(boxes_2d):
            cls_yolo = int(box.cls[0])
            if cls_yolo not in _COCO_TO_CLASS:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf       = float(box.conf[0])
            class_name = _COCO_TO_CLASS[cls_yolo]
            class_id   = _CLASS_TO_ID.get(class_name, 0)
            u_c        = (x1 + x2) / 2.0
            v_c        = (y1 + y2) / 2.0

            cx = cy = cz = w = h = l = ry = None
            n_pts  = 0
            source = "none"

            # Instance mask from seg model (bool [H,W]) or None → use box instead
            inst_mask: Optional[np.ndarray] = mask_imgs[det_i]

            # ── Method A: LiDAR + semantic filtering (primary) ────────────────
            if pts_cam is not None and us is not None:
                if inst_mask is not None:
                    # Tight per-instance mask — separates clustered vehicles
                    u_idx = np.clip(us.astype(np.int32), 0, W - 1)
                    v_idx = np.clip(vs.astype(np.int32), 0, H - 1)
                    in_region = inst_mask[v_idx, u_idx] & (pts_cam[:, 2] > 0.5)
                else:
                    # Fallback: rectangular bounding box
                    in_region = (
                        (us >= x1) & (us <= x2) &
                        (vs >= y1) & (vs <= y2) &
                        (pts_cam[:, 2] > 0.5)
                    )
                frustum = pts_cam[in_region]
                n_pts   = len(frustum)

                if sem_mask is not None and n_pts > 0:
                    # Keep only points landing on the correct semantic class
                    fu = np.clip(us[in_region].astype(np.int32), 0, W - 1)
                    fv = np.clip(vs[in_region].astype(np.int32), 0, H - 1)
                    sem_cls    = sem_mask[fv, fu]
                    target_ids = _CLASS_SEM_IDS.get(class_name, ())
                    sem_ok     = np.isin(sem_cls, target_ids)
                    if sem_ok.sum() >= 3:
                        frustum = frustum[sem_ok]
                        n_pts   = len(frustum)

                if n_pts >= self.min_frustum_pts:
                    cx, cy, cz, w, h, l, ry = _fit_box(frustum, class_name)
                    source = "lidar+seg" if inst_mask is not None else (
                        "lidar+sem" if sem_mask is not None else "lidar"
                    )

            # ── Method B: depth map fallback ─────────────────────────────────
            if cx is None and depth_map is not None:
                if inst_mask is not None:
                    # Use instance mask directly — tighter than box ROI
                    valid_px = inst_mask & (depth_map > 0.5) & (depth_map < 80.0)
                    if sem_mask is not None:
                        target_ids = _CLASS_SEM_IDS.get(class_name, ())
                        valid_px = valid_px & np.isin(sem_mask, target_ids)
                    vy, vx = np.where(valid_px)
                elif sem_mask is not None:
                    x1i = max(0, int(x1));  x2i = min(W - 1, int(x2))
                    y1i = max(0, int(y1));  y2i = min(H - 1, int(y2))
                    roi_sem = sem_mask [y1i:y2i + 1, x1i:x2i + 1]
                    roi_dep = depth_map[y1i:y2i + 1, x1i:x2i + 1].astype(np.float32)
                    target_ids = _CLASS_SEM_IDS.get(class_name, ())
                    valid_px   = (
                        np.isin(roi_sem, target_ids) &
                        (roi_dep > 0.5) & (roi_dep < 80.0)
                    )
                    vy_rel, vx_rel = np.where(valid_px)
                    vy = vy_rel + y1i
                    vx = vx_rel + x1i
                else:
                    vy = vx = np.array([], dtype=np.int32)

                if len(vy) >= self.min_depth_px:
                    Z     = depth_map[vy, vx].astype(np.float32)
                    X     = (vx - ppx) * Z / fx
                    Y     = (vy - ppy) * Z / fy
                    depth_pts = np.column_stack([X, Y, Z])
                    cx, cy, cz, w, h, l, ry = _fit_box(
                        depth_pts, class_name, skip_ground_filter=True
                    )
                    n_pts  = len(depth_pts)
                    source = "depth+seg" if inst_mask is not None else "depth+sem"

            # ── Method C: sparse LiDAR fallback ──────────────────────────────
            if cx is None and pts_cam is not None and us is not None:
                if inst_mask is not None:
                    u_idx = np.clip(us.astype(np.int32), 0, W - 1)
                    v_idx = np.clip(vs.astype(np.int32), 0, H - 1)
                    in_region = inst_mask[v_idx, u_idx] & (pts_cam[:, 2] > 0.5)
                else:
                    in_region = (
                        (us >= x1) & (us <= x2) &
                        (vs >= y1) & (vs <= y2) &
                        (pts_cam[:, 2] > 0.5)
                    )
                sparse = pts_cam[in_region]
                if len(sparse) > 0:
                    cz = float(np.median(sparse[:, 2]))
                    cx = (u_c - ppx) * cz / fx
                    cy = (v_c - ppy) * cz / fy
                    dw, dh, dl = _CLASS_DIMS.get(class_name, (1.5, 1.5, 3.0))
                    w, h, l, ry = dw, dh, dl, 0.0
                    n_pts  = len(sparse)
                    source = "lidar_sparse"

            if cx is None or cz is None or cz < 0.5:
                continue

            # ── Keypoint yaw override ─────────────────────────────────────────
            # Replace PCA ry with the front→rear direction computed from the
            # ApolloCar3D 24-keypoint model. Only applied to vehicle classes
            # where the model was trained (cars, trucks, buses).
            if kp_annotations and class_name in ("car", "truck", "bus") and depth_map is not None:
                kp_ry = self._kp_estimator.yaw_for_box(
                    kp_annotations,
                    [x1, y1, x2, y2],
                    cz,
                    depth_map,
                    intrinsics,
                    frame.shape,
                )
                if kp_ry is not None:
                    ry = kp_ry
                    source += "+kp"

            detections.append(Detection3D(
                frame_idx=frame_idx,
                class_id=class_id,
                class_name=class_name,
                score=conf,
                score_2d=conf,
                score_3d=min(1.0, n_pts / 50.0),
                box_2d=[x1, y1, x2, y2],
                box_3d=[cx, cy, cz, w, h, l, ry],
                centroid_2d=[u_c, v_c],
                centroid_3d=[cx, cy, cz],
                depth=cz,
            ))
            logger.debug(
                f"f{frame_idx} {class_name} z={cz:.1f}m "
                f"w={w:.1f} l={l:.1f} [{source}, {n_pts}pts]"
            )

        detections.sort(key=lambda d: d.score, reverse=True)

        # Cache for visualization (depth+keypoints, semantic+instances panels)
        self.last_kp_annotations = kp_annotations
        self.last_instance_masks = [
            (mask_imgs[i], int(box.cls[0]))
            for i, box in enumerate(boxes_2d)
            if mask_imgs[i] is not None
            and int(box.cls[0]) in _COCO_TO_CLASS
        ]

        return detections

    def unload(self) -> None:
        import torch
        self._model = None
        self._seg_model = None
        if self._kp_estimator:
            self._kp_estimator.unload()
        torch.cuda.empty_cache()
        logger.info("FrustumLidarDetector unloaded")
