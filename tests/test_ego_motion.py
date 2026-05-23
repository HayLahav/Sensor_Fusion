import numpy as np
from fusion_perception.tracking.ego_motion import estimate_homography, compensate_centroids

def test_identity_homography_on_same_frame():
    rng = np.random.default_rng(42)
    frame = (rng.random((370, 1224, 3)) * 255).astype(np.uint8)
    H = estimate_homography(frame, frame)
    # Identity or None (no motion)
    if H is not None:
        # applying H to a point should return same point
        pt = np.array([[[612., 185.]]], dtype=np.float32)
        import cv2
        warped = cv2.perspectiveTransform(pt, H)
        assert np.allclose(warped[0, 0], pt[0, 0], atol=2.0)

def test_compensate_centroids_no_motion():
    centroids = [[200., 150.], [400., 300.]]
    result = compensate_centroids(centroids, None)
    assert result == centroids

def test_compensate_centroids_with_H():
    import cv2
    H = np.eye(3, dtype=np.float32)
    H[0, 2] = 10.0   # pure translation of 10px in x
    centroids = [[100., 200.]]
    result = compensate_centroids(centroids, H)
    assert abs(result[0][0] - 110.0) < 1.0
    assert abs(result[0][1] - 200.0) < 1.0
