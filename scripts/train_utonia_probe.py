"""Train a linear probe on frozen Utonia features with KITTI-360 LiDAR labels.

Steps
  1. For every --stride-th labelled frame: load the scan, transfer labels from
     KITTI-360's accumulated semantic clouds (kitti360_semantics.py), and
     record the scan-to-map distance (calibration / pose check).
  2. Encode the scan with Utonia and cache a random subset of labelled points.
  3. Train a linear layer on the training frames (first part of the drive) and
     report mIoU on held-out frames (last --val-fraction of the drive, full
     scans, so the split does not share nearby frames).
  4. Save ckpt (probe + encoder settings) and a JSON report.

Example (Colab T4):
    python scripts/train_utonia_probe.py \\
        --kitti360-root /content/KITTI-360 \\
        --seq 2013_05_28_drive_0000_sync \\
        --variant utonia_small --stride 10 --max-frames 400 \\
        --out ckpt/utonia_probe.pt
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_perception.data.kitti360_semantics import Kitti360ScanLabeler, alignment_stats  # noqa: E402
from fusion_perception.data.lidar_loader import Kitti360Lidar, load_bin  # noqa: E402
from fusion_perception.segmentation.lidar_classes import CLASS_NAMES, IGNORE  # noqa: E402
from fusion_perception.segmentation.lidar_probe import (  # noqa: E402
    confusion_matrix, iou_from_confusion, predict_probe, save_probe, train_probe,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kitti360-root", required=True)
    p.add_argument("--seq", nargs="+", required=True, help="one or more training sequences")
    p.add_argument("--variant", default="utonia_small", choices=["utonia", "utonia_small", "utonia_tiny"])
    p.add_argument("--scale", type=float, default=0.2, help="Utonia coord scale (voxel = 0.01/scale m)")
    p.add_argument("--max-range", type=float, default=60.0)
    p.add_argument("--enable-flash", action="store_true", help="only on Ampere+ GPUs with flash-attn")
    p.add_argument("--stride", type=int, default=10, help="use every Nth labelled frame")
    p.add_argument("--max-frames", type=int, default=400, help="per sequence, after striding")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--points-per-frame", type=int, default=30000, help="training points kept per frame")
    p.add_argument("--max-dist", type=float, default=0.15, help="label transfer distance (m)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-dir", default="outputs/utonia_cache")
    p.add_argument("--out", default="ckpt/utonia_probe.pt")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from fusion_perception.models.utonia_wrapper import UtoniaEncoder

    enc_cfg = dict(variant=args.variant, scale=args.scale,
                   enable_flash=args.enable_flash, max_range=args.max_range)
    encoder = UtoniaEncoder(device=args.device, **enc_cfg)
    encoder.load()
    rng = np.random.default_rng(0)
    cache_dir = Path(args.cache_dir) / f"{args.variant}_s{args.scale}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_X, train_y = [], []
    val_items = []   # (seq, frame) evaluated on full scans later
    align: dict[str, list[dict]] = {}

    for seq in args.seq:
        labeler = Kitti360ScanLabeler(args.kitti360_root, seq, max_dist=args.max_dist)
        lidar = Kitti360Lidar.from_dir(os.path.join(args.kitti360_root, "data_3d_raw", seq))
        frames = labeler.frames_with_labels()[:: args.stride][: args.max_frames]
        n_val = max(1, int(len(frames) * args.val_fraction))
        split_at = len(frames) - n_val
        print(f"[{seq}] {len(frames)} frames: {split_at} train / {n_val} val")
        align[seq] = []

        for i, f in enumerate(frames):
            path = lidar.get_path(f)
            if path is None:
                continue
            cache = cache_dir / f"{seq}_{f:010d}.npz"
            if cache.exists():
                d = np.load(cache)
                feats, labels = d["feats"], d["labels"]
                xyz = d["xyz"]
                dist = d["dist"]
            else:
                xyz = load_bin(path)
                res = labeler.label_scan(xyz, f)
                if res is None:
                    continue
                labels, dist = res["labels"], res["dist"]
                feats = encoder.encode(xyz).astype(np.float16)
                np.savez_compressed(cache, feats=feats, labels=labels, xyz=xyz, dist=dist)

            stats = alignment_stats(dist, xyz)
            stats["frame"] = int(f)
            align[seq].append(stats)

            if i < split_at:
                ok = np.flatnonzero((labels != IGNORE) & encoder.range_mask(xyz))
                if len(ok) > args.points_per_frame:
                    ok = rng.choice(ok, args.points_per_frame, replace=False)
                train_X.append(feats[ok])
                train_y.append(labels[ok])
            else:
                val_items.append(cache)
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(frames)} frames, median scan-to-map "
                      f"{stats.get('median_m', float('nan')):.3f} m")

    if not train_X:
        print("No training frames found — check data_3d_semantics and data_3d_raw paths.")
        return 1

    X = np.concatenate(train_X)
    y = np.concatenate(train_y)
    print(f"Training probe on {len(y):,} points, feature dim {X.shape[1]}")
    probe = train_probe(X, y, epochs=args.epochs, device=args.device)

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for cache in val_items:
        d = np.load(cache)
        pred = predict_probe(probe, d["feats"], device=args.device)
        in_range = encoder.range_mask(d["xyz"])
        cm += confusion_matrix(pred[in_range], d["labels"][in_range])
    metrics = iou_from_confusion(cm)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta = {"encoder": enc_cfg, "class_names": CLASS_NAMES, "sequences": args.seq,
            "val_mIoU": metrics["mIoU"]}
    save_probe(args.out, probe, meta)

    align_summary = {
        seq: {
            "frames": len(v),
            "median_of_median_m": float(np.median([s["median_m"] for s in v if "median_m" in s]))
            if v else None,
            "mean_frac_within_15cm": float(np.mean([s["frac_within_15cm"] for s in v if "frac_within_15cm" in s]))
            if v else None,
        }
        for seq, v in align.items()
    }
    report = {"val": metrics, "alignment": align_summary, "alignment_per_frame": align, "meta": meta}
    report_path = Path(args.out).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nVal mIoU: {metrics['mIoU']:.3f}  accuracy: {metrics['accuracy']:.3f}")
    for name, iou in metrics["per_class_iou"].items():
        print(f"  {name:14s} {'—' if iou is None else f'{iou:.3f}'}")
    for seq, s in align_summary.items():
        print(f"[{seq}] scan-to-map median {s['median_of_median_m']:.3f} m, "
              f"{100 * s['mean_frac_within_15cm']:.0f}% of points within 15 cm")
    print(f"Saved probe → {args.out}\nReport → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
