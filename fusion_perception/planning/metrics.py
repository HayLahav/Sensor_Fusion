"""Open-loop planning metrics: planned path vs the path actually driven."""
from __future__ import annotations
import numpy as np


def displacement_errors(
    pred: np.ndarray,
    gt: np.ndarray,
    dt: float = 0.1,
    horizons_s: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> dict:
    """ADE / FDE (metres) at each horizon for [N,≥2] ego-frame paths.

    ADE@h: mean distance over waypoints up to h seconds.
    FDE@h: distance at the waypoint closest to h seconds.
    Horizons longer than either path are skipped.
    """
    pred = np.asarray(pred, dtype=np.float64)[:, :2]
    gt = np.asarray(gt, dtype=np.float64)[:, :2]
    n = min(len(pred), len(gt))
    d = np.linalg.norm(pred[:n] - gt[:n], axis=1)
    out: dict = {}
    for h in horizons_s:
        k = int(round(h / dt))
        if k < 1 or k > n:
            continue
        out[f"ADE@{h:g}s"] = float(d[:k].mean())
        out[f"FDE@{h:g}s"] = float(d[k - 1])
    return out
