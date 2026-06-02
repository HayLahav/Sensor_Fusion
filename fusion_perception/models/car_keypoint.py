"""Car keypoint detection via OpenPifPaf ApolloCar3D (24-keypoint model).

Checkpoint: shufflenetv2k16-apollo-24 (AP 76.1%, ~12 MB, auto-downloaded).

Keypoint indices (CAR_KEYPOINTS_24):
  0  front_up_right       1  front_up_left
  2  front_light_right    3  front_light_left
  4  front_low_right      5  front_low_left
  6  central_up_left      7  front_wheel_left
  8  rear_wheel_left      9  rear_corner_left
  10 rear_up_left         11 rear_up_right
  12 rear_light_left      13 rear_light_right
  14 rear_low_left        15 rear_low_right
  16 central_up_right     17 rear_corner_right
  18 rear_wheel_right     19 front_wheel_right
  20 rear_plate_left      21 rear_plate_right
  22 mirror_edge_left     23 mirror_edge_right

Yaw is arctan2(Δx, Δz) from the 3D centroid of front keypoints to rear keypoints,
back-projected via the depth map. Falls back to None when fewer than 2 keypoints
survive confidence filtering on either end.
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("car_keypoint")

_CHECKPOINT = "shufflenetv2k16-apollo-24"
_CONF_THRESHOLD = 0.15   # minimum keypoint confidence to use for 3D projection

# Front face of the car: corners, lights, low bumper, front wheels
_FRONT_KPS = [0, 1, 2, 3, 4, 5, 7, 19]
# Rear face: corners, lights, low bumper, rear wheels, plate
_REAR_KPS  = [8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21]


def _box_iou(a: list[float], b: list[float]) -> float:
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / (union + 1e-6)


def _backproject_keypoints(
    kps: np.ndarray,           # [K, 3] float  — (x, y, conf) in image pixels
    indices: list[int],
    depth_map: np.ndarray,     # [H, W] float32 metric depth (metres)
    fx: float, fy: float,
    ppx: float, ppy: float,
    H: int, W: int,
) -> np.ndarray:
    """Return [N, 3] camera-frame 3D points for keypoints with conf > threshold."""
    pts: list[list[float]] = []
    for idx in indices:
        if idx >= len(kps):
            continue
        x, y, c = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
        if c < _CONF_THRESHOLD:
            continue
        xi = int(np.clip(x, 0, W - 1))
        yi = int(np.clip(y, 0, H - 1))
        Z  = float(depth_map[yi, xi])
        if Z < 0.5 or Z > 80.0:
            continue
        pts.append([(x - ppx) * Z / fx, (y - ppy) * Z / fy, Z])
    return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)


class CarKeypointEstimator:
    """Wraps OpenPifPaf ApolloCar3D to produce per-vehicle yaw angles.

    Usage inside FrustumLidarDetector.detect():
        anns = self._kp_estimator.predict_frame(frame)
        for det in detections:
            ry = self._kp_estimator.yaw_for_box(
                anns, det.box_2d, det.depth, depth_map, intrinsics, frame.shape
            )
            if ry is not None:
                det.box_3d[6] = ry
    """

    def __init__(
        self,
        checkpoint: str = _CHECKPOINT,
        device: str = "cuda",
    ) -> None:
        self._checkpoint = checkpoint
        self._device     = device
        self._predictor  = None

    def load(self) -> None:
        try:
            import openpifpaf
            import openpifpaf.plugins.apollocar3d  # registers the car plugin
            self._predictor = openpifpaf.Predictor(
                checkpoint=self._checkpoint,
                device=self._device,
            )
            logger.info(f"CarKeypointEstimator loaded: {self._checkpoint}")
            print(f"CarKeypointEstimator loaded: {self._checkpoint}")
        except ImportError:
            self._predictor = None
            print(
                "WARNING: openpifpaf not installed — car keypoints disabled.\n"
                "  Fix: add  !pip install openpifpaf -q  to the Colab install cell."
            )
        except Exception as exc:
            self._predictor = None
            print(f"WARNING: CarKeypointEstimator failed to load ({exc}) — keypoints disabled.")

    @property
    def available(self) -> bool:
        return self._predictor is not None

    def predict_frame(self, frame_rgb: np.ndarray) -> list:
        """Run keypoint detection on a full frame. Returns list of Annotation objects."""
        if not self.available:
            return []
        try:
            from PIL import Image
            pil = Image.fromarray(frame_rgb)
            predictions, _, _ = self._predictor.pil_image(pil)
            return predictions
        except Exception as exc:
            logger.debug(f"CarKeypointEstimator.predict_frame failed: {exc}")
            return []

    def yaw_for_box(
        self,
        annotations: list,
        box_2d: list[float],        # [x1, y1, x2, y2] of the detection
        det_cz: float,              # detection depth (fallback if no depth_map)
        depth_map: Optional[np.ndarray],
        intrinsics: Optional[np.ndarray],
        frame_shape: tuple[int, int, int],
    ) -> Optional[float]:
        """Return keypoint-based yaw for the detection, or None on failure.

        Matches the best-IoU annotation to `box_2d`, then back-projects front
        and rear keypoints to 3D and computes arctan2(Δx, Δz).
        """
        if not annotations or depth_map is None or intrinsics is None:
            return None

        H, W = frame_shape[:2]
        fx  = float(intrinsics[0, 0]); fy  = float(intrinsics[1, 1])
        ppx = float(intrinsics[0, 2]); ppy = float(intrinsics[1, 2])

        # ── Match annotation to detection box by IoU ─────────────────────────
        best_ann  = None
        best_iou  = 0.25   # minimum IoU to consider a match
        for ann in annotations:
            try:
                bx, by, bw, bh = ann.bbox()
                ann_box = [bx, by, bx + bw, by + bh]
            except Exception:
                continue
            iou = _box_iou(box_2d, ann_box)
            if iou > best_iou:
                best_iou = iou
                best_ann = ann

        if best_ann is None:
            return None

        kps = np.array(best_ann.data, dtype=np.float32)  # [24, 3] (x, y, conf)
        if kps.shape[0] < 24:
            return None

        # ── Back-project front and rear keypoints to 3D ──────────────────────
        front_3d = _backproject_keypoints(kps, _FRONT_KPS, depth_map, fx, fy, ppx, ppy, H, W)
        rear_3d  = _backproject_keypoints(kps, _REAR_KPS,  depth_map, fx, fy, ppx, ppy, H, W)

        if len(front_3d) < 2 or len(rear_3d) < 2:
            return None

        front_c = front_3d.mean(axis=0)   # [X, Y, Z]
        rear_c  = rear_3d.mean(axis=0)

        dv = front_c - rear_c             # front→rear direction vector in camera frame
        if abs(dv[0]) < 1e-3 and abs(dv[2]) < 1e-3:
            return None

        ry = float(np.arctan2(dv[0], dv[2]))
        logger.debug(
            f"Keypoint yaw: front_c={front_c.round(2)} rear_c={rear_c.round(2)} ry={np.degrees(ry):.1f}°"
        )
        return ry

    def unload(self) -> None:
        self._predictor = None
