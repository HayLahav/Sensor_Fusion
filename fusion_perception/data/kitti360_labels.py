"""KITTI-360 GT 3D bounding-box loader.

Requires kitti360scripts (pip install kitti360scripts) and the following
dataset files relative to dataset_root:
  data_3d_bboxes/{train|test}/{sequence}.xml  — 3D box annotations
  data_poses/{sequence}/poses.txt             — per-frame ego poses
  calibration/calib_cam_to_pose.txt           — cam00 → IMU/pose transform
  calibration/perspective.txt                 — R_rect_00

Usage:
    labels = Kitti360GTLabels(dataset_root, sequence)
    boxes  = labels.get_boxes_for_frame(frame_idx)
    # boxes: list of {'class': str, 'box3d': [cx,cy,cz,h,l,w,qw,qx,qy,qz],
    #                  'instance_id': int}
"""
from __future__ import annotations
import os
import numpy as np
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("kitti360_labels")

DYNAMIC_CLASSES = frozenset([
    "car", "truck", "bicycle", "motorcyclist", "pedestrian", "rider",
])


def _load_poses(pose_file: str) -> dict[int, np.ndarray]:
    """poses.txt → {frame_idx: 4×4 T_imu_to_world (float64)}."""
    poses: dict[int, np.ndarray] = {}
    with open(pose_file) as f:
        for line in f:
            vals = line.strip().split()
            if not vals:
                continue
            frame_idx = int(vals[0])
            mat = np.array(vals[1:], dtype=np.float64).reshape(3, 4)
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = mat
            poses[frame_idx] = T
    return poses


def _rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation matrix → (qw, qx, qy, qz)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def _obj_to_box3d(obj, T_world_to_rectcam: np.ndarray) -> list[float]:
    """Convert KITTI360Bbox3D to [cx,cy,cz,h,l,w,qw,qx,qy,qz] in camera frame.

    obj.R / obj.T describe the local→world transform.
    obj.vertices are the 8 corners already in world frame.

    Dimensions are axis-aligned extents in the rectified camera frame
    (x=lateral, y=down, z=depth/forward), matching what bev_iou expects.
    KITTI-360 object-local frame uses x=forward, y=left, z=up — a different
    axis convention — so we project the 8 world-frame corners into camera
    frame and take the AABB there rather than relying on local-frame indices.
    """
    # Center in camera frame
    center_h = np.array([obj.T[0], obj.T[1], obj.T[2], 1.0], dtype=np.float64)
    cx, cy, cz = (T_world_to_rectcam @ center_h)[:3]

    # Rotation in camera frame (for yaw)
    R_cam = (T_world_to_rectcam[:3, :3] @ obj.R).astype(np.float64)

    # Camera-frame axis-aligned extents from the 8 world-frame corners
    verts_h = np.hstack([obj.vertices, np.ones((len(obj.vertices), 1))])
    verts_cam = (T_world_to_rectcam @ verts_h.T).T[:, :3]
    w = float(verts_cam[:, 0].max() - verts_cam[:, 0].min())  # camera x = lateral
    h = float(verts_cam[:, 1].max() - verts_cam[:, 1].min())  # camera y = height
    l = float(verts_cam[:, 2].max() - verts_cam[:, 2].min())  # camera z = depth

    qw, qx, qy, qz = _rot_to_quat(R_cam)
    return [float(cx), float(cy), float(cz), h, l, w, qw, qx, qy, qz]


class Kitti360GTLabels:
    """Load KITTI-360 GT 3D bounding boxes and transform to rectified camera frame.

    Parameters
    ----------
    dataset_root : str
        Root directory of the KITTI-360 dataset (contains data_3d_bboxes/,
        data_poses/, calibration/).
    sequence : str
        Sequence name, e.g. ``2013_05_28_drive_0000_sync``.
    calib_dir : str | None
        Path to calibration/ directory.  Defaults to ``{dataset_root}/calibration``.
    classes : frozenset[str] | None
        Which semantic classes to include.  Defaults to DYNAMIC_CLASSES.
    """

    def __init__(
        self,
        dataset_root: str,
        sequence: str,
        calib_dir: str | None = None,
        classes: frozenset[str] | None = None,
    ) -> None:
        from kitti360scripts.helpers.annotation import Annotation3D
        from kitti360scripts.devkits.commons.loadCalibration import (
            loadCalibrationCameraToPose, loadPerspectiveIntrinsic,
        )

        if calib_dir is None:
            calib_dir = os.path.join(dataset_root, "calibration")
        self._classes = classes if classes is not None else DYNAMIC_CLASSES

        label_dir = os.path.join(dataset_root, "data_3d_bboxes")
        # Annotation3D globs labelDir/*/seq.xml and errors if >1 match.
        # data_3d_bboxes.zip ships both train/ and train_full/ — resolve to train/.
        import glob as _glob, tempfile, shutil
        xml_matches = _glob.glob(os.path.join(label_dir, "*", f"{sequence}.xml"))
        if len(xml_matches) != 1:
            preferred = os.path.join(label_dir, "train", f"{sequence}.xml")
            if not os.path.isfile(preferred):
                raise FileNotFoundError(
                    f"No unique annotation found for {sequence} in {label_dir}. "
                    f"Matches: {xml_matches}"
                )
            _tmp = tempfile.mkdtemp()
            try:
                _sub = os.path.join(_tmp, "train")
                os.makedirs(_sub)
                shutil.copy(preferred, os.path.join(_sub, f"{sequence}.xml"))
                self.annotation = Annotation3D(_tmp, sequence)
            finally:
                shutil.rmtree(_tmp, ignore_errors=True)
        else:
            self.annotation = Annotation3D(label_dir, sequence)

        pose_file = os.path.join(dataset_root, "data_poses", sequence, "poses.txt")
        self.poses = _load_poses(pose_file)

        Tr = loadCalibrationCameraToPose(os.path.join(calib_dir, "calib_cam_to_pose.txt"))
        T_cam_to_pose = Tr["image_00"].astype(np.float64)
        self._T_pose_to_cam = np.linalg.inv(T_cam_to_pose)

        intrinsics = loadPerspectiveIntrinsic(os.path.join(calib_dir, "perspective.txt"))
        self._R_rect = intrinsics["R_rect_00"].astype(np.float64)

        # Expand R_rect to 4×4 once
        self._R_rect_4x4 = np.eye(4, dtype=np.float64)
        self._R_rect_4x4[:3, :3] = self._R_rect

        logger.info(
            f"Kitti360GTLabels: sequence={sequence}, "
            f"{self.annotation.num_bbox} boxes, {len(self.poses)} pose frames"
        )

    def get_boxes_for_frame(self, frame_idx: int) -> list[dict]:
        """Return GT boxes visible at *frame_idx* in the rectified camera frame.

        Returns
        -------
        list of dict with keys:
            ``class``       : semantic class name (str)
            ``box3d``       : [cx, cy, cz, h, l, w, qw, qx, qy, qz]
            ``instance_id`` : instance ID (int)
        """
        if frame_idx not in self.poses:
            logger.debug(f"No pose for frame {frame_idx}")
            return []

        T_imu_to_world = self.poses[frame_idx]
        T_world_to_imu = np.linalg.inv(T_imu_to_world)
        # world → IMU/pose → unrectified cam → rectified cam
        T_world_to_rectcam = self._R_rect_4x4 @ self._T_pose_to_cam @ T_world_to_imu

        boxes: list[dict] = []
        for timestamps_dict in self.annotation.objects.values():
            for ts, obj in timestamps_dict.items():
                if ts == -1:
                    if not (obj.start_frame <= frame_idx <= obj.end_frame):
                        continue
                elif ts != frame_idx:
                    continue

                if obj.name not in self._classes:
                    continue

                # Skip objects entirely behind the camera
                verts_h = np.hstack([obj.vertices, np.ones((len(obj.vertices), 1))])
                verts_cam = (T_world_to_rectcam @ verts_h.T).T[:, :3]
                if verts_cam[:, 2].max() <= 0.1:
                    continue

                box3d = _obj_to_box3d(obj, T_world_to_rectcam)
                boxes.append({
                    "class": obj.name,
                    "box3d": box3d,
                    "instance_id": int(obj.instanceId),
                })

        return boxes

    # Map KITTI-360 class names to detection-prompt names for mAP comparison.
    _CLASS_ALIAS: dict[str, str] = {
        "pedestrian": "person",
        "motorcyclist": "cyclist",
        "rider": "cyclist",
        "bicycle": "cyclist",   # KITTI-360 annotates bicycle (vehicle) separately
    }

    def to_gt_labels(self, frame_idx: int) -> list:
        """Return GT boxes as GTLabel objects for use with metrics.py.

        Converts from internal [cx,cy,cz,h,l,w,qw,qx,qy,qz] to the
        GTLabel format [cx,cy,cz,w,h,l,ry] expected by bev_iou / compute_*_metrics.
        Class names are normalised to match detector prompt names via _CLASS_ALIAS.
        """
        import math
        from fusion_perception.utils.dataclasses import GTLabel
        result = []
        for b in self.get_boxes_for_frame(frame_idx):
            cx, cy, cz, h, l, w = b["box3d"][:6]
            qw, qx, qy, qz = b["box3d"][6:]
            # yaw around y-axis in camera frame
            ry = math.atan2(2.0 * (qw * qy - qx * qz),
                            1.0 - 2.0 * (qy * qy + qz * qz))
            cls = self._CLASS_ALIAS.get(b["class"], b["class"])
            result.append(GTLabel(
                track_id=b["instance_id"],
                class_name=cls,
                box_3d=[cx, cy, cz, float(w), float(h), float(l), float(ry)],
            ))
        return result

    def get_training_entries(
        self,
        frame_idx: int,
        image_path: str,
    ) -> list[dict]:
        """Return structured training entries suitable for Gemma fine-tuning.

        Each entry contains the GT box list for a single frame as a JSON-
        serialisable dict with keys ``image_path``, ``frame_idx``, and
        ``gt_boxes``.
        """
        boxes = self.get_boxes_for_frame(frame_idx)
        return {
            "image_path": image_path,
            "frame_idx": frame_idx,
            "gt_boxes": boxes,
        }
