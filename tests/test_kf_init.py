import numpy as np
from fusion_perception.tracking.kf_init import init_kf, KF_DIM_X, KF_DIM_Z

def test_kf_dimensions():
    box3d = [1.0, 0.0, 15.0, 0.0, 4.0, 2.0, 1.5]
    kf = init_kf(box3d, dt=0.1)
    assert kf.x.shape == (KF_DIM_X, 1)
    assert kf.F.shape == (KF_DIM_X, KF_DIM_X)
    assert kf.H.shape == (KF_DIM_Z, KF_DIM_X)

def test_kf_state_initialized():
    box3d = [1.0, 0.0, 15.0, 0.3, 4.0, 2.0, 1.5]
    kf = init_kf(box3d, dt=0.1)
    x = kf.x.flatten()
    assert abs(x[0] - 1.0) < 1e-6   # cx
    assert abs(x[2] - 15.0) < 1e-6  # cz
    assert abs(x[3] - 0.3) < 1e-6   # theta
    # velocities should start at zero
    np.testing.assert_array_equal(x[7:], np.zeros(3))

def test_kf_predict_advances_position():
    box3d = [0., 0., 10., 0., 4., 2., 1.5]
    kf = init_kf(box3d, dt=0.1)
    # inject velocity
    kf.x[7, 0] = 5.0  # vx = 5 m/s
    kf.predict()
    x = kf.x.flatten()
    assert abs(x[0] - 0.5) < 1e-5   # cx advanced by vx*dt = 0.5

def test_kf_transition_matrix_velocity():
    box3d = [0., 0., 10., 0., 4., 2., 1.5]
    kf = init_kf(box3d, dt=0.2)
    assert abs(kf.F[0, 7] - 0.2) < 1e-9  # cx row, vx col
    assert abs(kf.F[1, 8] - 0.2) < 1e-9
    assert abs(kf.F[2, 9] - 0.2) < 1e-9
