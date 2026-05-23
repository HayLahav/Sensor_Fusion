"""Ego-motion compensation via ORB background homography (BoT-SORT style)."""
from __future__ import annotations
from typing import Optional
import cv2
import numpy as np


def estimate_homography(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    max_features: int = 200,
) -> Optional[np.ndarray]:
    """
    Estimate 3×3 homography from background ORB matches between two frames.
    Returns None if too few inliers are found.
    frames must be RGB, shape (H,W,3) uint8 — the pipeline standard used throughout the tracker.
    """
    orb = cv2.ORB_create(nfeatures=max_features)
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_RGB2GRAY)

    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn_matches if len([m, n]) == 2 and m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H is None:
        return None
    inliers = int(mask.sum()) if mask is not None else 0
    return H if inliers >= 15 else None


def compensate_centroids(
    centroids: list[list[float]],
    H: Optional[np.ndarray],
) -> list[list[float]]:
    """Apply homography H to a list of [x, y] pixel centroids."""
    if H is None or len(centroids) == 0:
        return [list(c) for c in centroids]

    pts = np.array(centroids, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    return warped.reshape(-1, 2).tolist()
