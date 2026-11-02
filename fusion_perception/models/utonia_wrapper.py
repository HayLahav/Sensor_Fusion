"""Utonia point-cloud encoder wrapper (Pointcept/Utonia, PTv3-based).

Utonia is encoder-only: it returns one feature vector per LiDAR point. A
linear probe (segmentation/lidar_probe.py) turns those features into the
semantic classes in segmentation/lidar_classes.py.

Install (CUDA required):
    pip install spconv-cu121
    pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
    pip install git+https://github.com/Pointcept/Utonia
    # optional, not on T4: pip install flash-attn

Licence note: Utonia code is Apache-2.0, but the pretrained weights are
CC-BY-NC 4.0 (non-commercial use only).

Input frame: raw Velodyne coordinates (x forward, y left, z up, sensor at the
origin), which is what Utonia's outdoor setting expects (road on the xy-plane,
apply_z_positive=False).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("utonia_wrapper")


class UtoniaEncoder:
    """Per-point features for one LiDAR scan.

    Parameters
    ----------
    variant : "utonia" | "utonia_small" | "utonia_tiny"
    scale   : coordinate scale applied before the 0.01 grid sampling, so the
              effective voxel size is 0.01 / scale metres. 0.2 → 5 cm voxels.
              Utonia's README says this needs tuning per domain.
    enable_flash : FlashAttention; must be False on T4 / pre-Ampere GPUs.
    max_range : drop points farther than this (metres) before encoding.
    """

    def __init__(
        self,
        variant: str = "utonia_small",
        repo_id: str = "Pointcept/Utonia",
        scale: float = 0.2,
        enable_flash: bool = False,
        max_range: float = 60.0,
        device: str = "cuda",
    ) -> None:
        self.variant = variant
        self.repo_id = repo_id
        self.scale = scale
        self.enable_flash = enable_flash
        self.max_range = max_range
        self.device = device
        self._model = None
        self._transform = None
        self.feat_dim: Optional[int] = None

    def load(self) -> None:
        import utonia  # noqa: F401 — imported lazily so the package stays optional

        custom_config = dict(enable_flash=self.enable_flash)
        try:
            model = utonia.model.load(
                self.variant, repo_id=self.repo_id, custom_config=custom_config
            )
        except TypeError:
            # Older releases without custom_config
            model = utonia.model.load(self.variant, repo_id=self.repo_id)
        self._model = model.to(self.device).eval()
        self._transform = utonia.transform.default(
            self.scale, apply_z_positive=False
        )
        logger.info(
            f"Utonia '{self.variant}' loaded (scale={self.scale}, "
            f"flash={self.enable_flash}, device={self.device})"
        )

    def range_mask(self, xyz: np.ndarray) -> np.ndarray:
        return np.linalg.norm(xyz[:, :2], axis=1) <= self.max_range

    def encode(
        self,
        xyz_velo: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """[N,3] Velodyne points → [N,C] float32 features.

        Points beyond max_range get zero features; use range_mask() to skip them.
        colors: optional [N,3] uint8 RGB (e.g. projected from the camera);
        zeros are used otherwise, as the Utonia README recommends.
        """
        import torch

        if self._model is None:
            raise RuntimeError("UtoniaEncoder.load() must be called first")

        keep = self.range_mask(xyz_velo)
        xyz = xyz_velo[keep].astype(np.float32)
        n = len(xyz)
        col = (np.zeros((n, 3), dtype=np.float32) if colors is None
               else colors[keep].astype(np.float32))
        point = {
            "coord": xyz,
            "color": col,
            "normal": np.zeros((n, 3), dtype=np.float32),
        }
        point = self._transform(point)
        for k, v in point.items():
            if isinstance(v, torch.Tensor):
                point[k] = v.to(self.device, non_blocking=True)

        with torch.inference_mode():
            point = self._model(point)
            point = _upcast_to_full_resolution(point)
            feat = point.feat[point.inverse]   # back to every input point

        feat_np = feat.float().cpu().numpy()
        self.feat_dim = feat_np.shape[1]
        out = np.zeros((len(xyz_velo), self.feat_dim), dtype=np.float32)
        out[keep] = feat_np
        return out

    def unload(self) -> None:
        self._model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass


def _upcast_to_full_resolution(point):
    """Walk the PTv3 pooling hierarchy back to the finest level.

    The first two levels concatenate coarse features onto the finer ones
    (multi-scale features, as in the Sonata/Utonia demos); deeper levels just
    copy features down.
    """
    import torch

    for _ in range(2):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent
    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent
    return point
