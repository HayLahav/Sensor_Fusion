"""Kalman Filter factory for KalmanCoWTracker."""
import numpy as np
from filterpy.kalman import KalmanFilter

KF_DIM_X = 10   # [cx, cy, cz, θ, l, w, h, vx, vy, vz]
KF_DIM_Z = 7    # [cx, cy, cz, θ, l, w, h]

# Index map: state vector positions
IDX_CX, IDX_CY, IDX_CZ = 0, 1, 2
IDX_THETA = 3
IDX_L, IDX_W, IDX_H = 4, 5, 6
IDX_VX, IDX_VY, IDX_VZ = 7, 8, 9


def init_kf(box3d: list[float], dt: float = 0.1) -> KalmanFilter:
    """
    Create and initialise a KalmanFilter for one track.
    box3d: [cx, cy, cz, theta, l, w, h]
    dt: time step in seconds (1/FPS)

    Note: F[CX,VX], F[CY,VY], F[CZ,VZ] are overwritten before each predict() call
    by the tracker loop to support variable fps. The `dt` here sets only the
    initial value; callers should update F.dt entries before calling kf.predict().
    """
    kf = KalmanFilter(dim_x=KF_DIM_X, dim_z=KF_DIM_Z)

    # State transition: constant velocity for position, identity for rest
    kf.F = np.eye(KF_DIM_X, dtype=np.float64)
    kf.F[IDX_CX, IDX_VX] = dt
    kf.F[IDX_CY, IDX_VY] = dt
    kf.F[IDX_CZ, IDX_VZ] = dt

    # Observation: directly observe first 7 state dims
    kf.H = np.zeros((KF_DIM_Z, KF_DIM_X), dtype=np.float64)
    kf.H[:KF_DIM_Z, :KF_DIM_Z] = np.eye(KF_DIM_Z)

    # Process noise: higher on velocities (sudden braking/turning)
    kf.Q = np.diag([
        0.1, 0.1, 0.2,       # cx, cy, cz
        0.05,                 # theta
        0.02, 0.02, 0.02,    # l, w, h
        2.0, 2.0, 3.0,       # vx, vy, vz  (3.0 for forward speed changes)
    ]).astype(np.float64)

    # Initial state covariance: high uncertainty on velocities
    kf.P = np.diag([
        1.0, 1.0, 2.0,
        0.1,
        0.5, 0.5, 0.5,
        10.0, 10.0, 10.0,
    ]).astype(np.float64)

    # Measurement noise placeholder (overridden per-frame by synthesize_measurement)
    kf.R = np.diag([0.5, 0.5, 2.0, 0.1, 0.3, 0.3, 0.3]).astype(np.float64)

    # Initial state
    kf.x = np.zeros((KF_DIM_X, 1), dtype=np.float64)
    kf.x[:KF_DIM_Z, 0] = box3d

    return kf
