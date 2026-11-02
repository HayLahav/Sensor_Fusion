"""Linear probe on frozen Utonia features → LiDAR semantic classes.

Training data comes from scripts/train_utonia_probe.py (features + labels
transferred from KITTI-360's accumulated semantic point clouds).
"""
from __future__ import annotations
from typing import Optional
import numpy as np
from fusion_perception.segmentation.lidar_classes import (
    CLASS_NAMES, IGNORE, NUM_CLASSES,
)
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("lidar_probe")


def confusion_matrix(pred: np.ndarray, gt: np.ndarray, n: int = NUM_CLASSES) -> np.ndarray:
    ok = gt != IGNORE
    idx = gt[ok].astype(np.int64) * n + pred[ok].astype(np.int64)
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def iou_from_confusion(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    denom = cm.sum(0) + cm.sum(1) - tp
    present = cm.sum(1) > 0
    iou = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    per_class = {CLASS_NAMES[i]: (float(iou[i]) if present[i] else None)
                 for i in range(len(tp))}
    miou = float(np.nanmean(iou[present])) if present.any() else float("nan")
    acc = float(tp.sum() / max(cm.sum(), 1))
    return {"mIoU": miou, "accuracy": acc, "per_class_iou": per_class}


def build_probe(in_dim: int, num_classes: int = NUM_CLASSES):
    import torch.nn as nn
    return nn.Linear(in_dim, num_classes)


def train_probe(
    feats: np.ndarray,
    labels: np.ndarray,
    epochs: int = 20,
    batch_size: int = 65536,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    class_balanced: bool = True,
    device: str = "cuda",
    seed: int = 0,
):
    """Train a linear layer on [M,C] features and [M] labels (IGNORE skipped)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    keep = labels != IGNORE
    X = torch.from_numpy(feats[keep].astype(np.float32))
    y = torch.from_numpy(labels[keep].astype(np.int64))
    if len(y) == 0:
        raise ValueError("No labelled points to train on")

    weight = None
    if class_balanced:
        counts = np.bincount(y.numpy(), minlength=NUM_CLASSES).astype(np.float64)
        w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
        w = w / w[counts > 0].mean()
        weight = torch.tensor(w, dtype=torch.float32, device=device)

    probe = build_probe(X.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx].to(device), y[idx].to(device)
            loss = loss_fn(probe(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()
        logger.info(f"probe epoch {ep + 1}/{epochs} loss={total / n:.4f}")
    return probe.eval()


def predict_probe(probe, feats: np.ndarray, device: str = "cuda",
                  batch_size: int = 262144) -> np.ndarray:
    import torch

    out = np.empty(len(feats), dtype=np.uint8)
    with torch.inference_mode():
        for i in range(0, len(feats), batch_size):
            xb = torch.from_numpy(feats[i:i + batch_size].astype(np.float32)).to(device)
            out[i:i + batch_size] = probe(xb).argmax(1).cpu().numpy().astype(np.uint8)
    return out


def save_probe(path: str, probe, meta: dict) -> None:
    import torch
    torch.save({"state_dict": probe.state_dict(), "meta": meta}, path)


def load_probe(path: str, device: str = "cuda"):
    import torch
    ckpt = torch.load(path, map_location=device)
    w = ckpt["state_dict"]["weight"]
    probe = build_probe(w.shape[1], w.shape[0]).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    return probe.eval(), ckpt.get("meta", {})


class LidarSemanticSegmenter:
    """Utonia encoder + trained linear probe → per-point class ids."""

    def __init__(self, probe_path: str, device: str = "cuda",
                 encoder_overrides: Optional[dict] = None) -> None:
        self.probe_path = probe_path
        self.device = device
        self.encoder_overrides = encoder_overrides or {}
        self.encoder = None
        self.probe = None

    def load(self) -> None:
        from fusion_perception.models.utonia_wrapper import UtoniaEncoder

        self.probe, meta = load_probe(self.probe_path, self.device)
        enc_cfg = dict(meta.get("encoder", {}))
        enc_cfg.update(self.encoder_overrides)   # e.g. enable_flash on newer GPUs
        self.encoder = UtoniaEncoder(device=self.device, **enc_cfg)
        self.encoder.load()
        logger.info(f"LiDAR segmenter ready (probe: {self.probe_path}, encoder: {enc_cfg})")

    def predict(self, xyz_velo: np.ndarray) -> np.ndarray:
        """[N,3] Velodyne points → [N] class ids (IGNORE beyond encoder range)."""
        feats = self.encoder.encode(xyz_velo)
        labels = predict_probe(self.probe, feats, self.device)
        labels[~self.encoder.range_mask(xyz_velo)] = IGNORE
        return labels
