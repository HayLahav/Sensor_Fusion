"""Cost matrix, Mahalanobis gate, and Hungarian matching for KalmanCoWTracker."""
from __future__ import annotations
import numpy as np
from fusion_perception.utils.geometry import iou3d

try:
    from lapjv import lapjv as _lapjv
    _HAS_LAPJV = True
except ImportError:
    from scipy.optimize import linear_sum_assignment as _scipy_lsa
    _HAS_LAPJV = False


def build_cost_matrix(
    pred_boxes: list[list[float]],   # [N] predicted boxes [cx,cy,cz,θ,l,w,h]
    det_boxes: list[list[float]],    # [M] detected boxes  [cx,cy,cz,θ,l,w,h]
    cow_valid: set[int],             # unused — kept for API compatibility
    alpha: float = 0.35,
) -> np.ndarray:
    """
    Hybrid cost: α·(1−IoU3D) + (1−α)·3D_distance_normalised.
    Shape: [N, M]

    3D centre distance (normalised to 20 m) is always used as the motion term.
    IoU3D is effectively zero whenever monocular depth estimates jump between
    frames, so using it as the sole fallback cost meant cost=1.0 for the same
    object → threshold rejection → track churn.  The distance term is robust
    to depth noise and provides a meaningful signal even without CoWTracker.
    """
    N, M = len(pred_boxes), len(det_boxes)
    if N == 0 or M == 0:
        return np.zeros((N, M))

    C = np.ones((N, M), dtype=np.float64)
    det_centers = np.array([[d[0], d[1], d[2]] for d in det_boxes])

    for i, pb in enumerate(pred_boxes):
        pb_center = np.array([pb[0], pb[1], pb[2]])

        for j, db in enumerate(det_boxes):
            iou = iou3d(pb, db)
            iou_cost = 1.0 - iou

            dc = float(np.linalg.norm(pb_center - det_centers[j]))
            dist_cost = min(dc / 20.0, 1.0)

            C[i, j] = alpha * iou_cost + (1.0 - alpha) * dist_cost

    return C


def mahalanobis_gate(
    pred_states: list[np.ndarray],    # [N] each shape (10,)
    pred_covs: list[np.ndarray],      # [N] each shape (10,10)
    det_z: list[np.ndarray],          # [M] each shape (7,)
    threshold: float = 13.28,         # χ²(0.99, df=4) — gates on [cx,cy,cz,θ]
) -> np.ndarray:
    """Boolean mask [N, M]: True = association is plausible."""
    N, M = len(pred_states), len(det_z)
    mask = np.ones((N, M), dtype=bool)
    H = np.zeros((7, 10))
    H[:7, :7] = np.eye(7)

    for i in range(N):
        x = pred_states[i]
        P = pred_covs[i]
        S = H @ P @ H.T
        S_sub = S[:4, :4]  # gate on [cx, cy, cz, θ] only (4 dof)
        try:
            S_inv = np.linalg.inv(S_sub + np.eye(4) * 1e-6)
        except np.linalg.LinAlgError:
            continue
        pred_obs = x[:7]
        for j in range(M):
            diff = det_z[j][:4] - pred_obs[:4]
            d2 = float(diff @ S_inv @ diff)
            mask[i, j] = d2 <= threshold

    return mask


def hungarian_match(
    cost_matrix: np.ndarray,
    threshold: float = 0.5,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Solve assignment problem. Returns (matched, unmatched_rows, unmatched_cols).
    matched: list of (row_idx, col_idx) pairs with cost < threshold
    """
    N, M = cost_matrix.shape
    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    if _HAS_LAPJV:
        size = max(N, M)
        padded = np.ones((size, size), dtype=np.float64)
        padded[:N, :M] = cost_matrix
        row_ind, _col_ind, _ = _lapjv(padded)
        # row_ind[r] = column assigned to row r
        assignment = [(r, int(row_ind[r])) for r in range(N) if int(row_ind[r]) < M]
    else:
        row_ind, col_ind = _scipy_lsa(cost_matrix)
        assignment = list(zip(row_ind.tolist(), col_ind.tolist()))

    matched = [(r, c) for r, c in assignment if cost_matrix[r, c] <= threshold]
    matched_rows = {r for r, _ in matched}
    matched_cols = {c for _, c in matched}
    unmatched_rows = [r for r in range(N) if r not in matched_rows]
    unmatched_cols = [c for c in range(M) if c not in matched_cols]

    return matched, unmatched_rows, unmatched_cols
