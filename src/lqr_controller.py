# src/lqr_controller.py

import numpy as np
import control

# Spacecraft inertia matrix
I = np.diag([0.02, 0.025, 0.03])

# Linearized spacecraft rotational dynamics
# State vector:
# x = [q_error_x, q_error_y, q_error_z,
#      wx, wy, wz]

A = np.block([
    [np.zeros((3, 3)), 0.5 * np.eye(3)],
    [np.zeros((3, 3)), np.zeros((3, 3))]
])

B = np.block([
    [np.zeros((3, 3))],
    [np.linalg.inv(I)]
])

# Weight matrices
Q = np.diag([120, 120, 120, 15, 15, 15])
R = np.diag([0.1, 0.1, 0.1])

# Solve LQR gain
K, _, _ = control.lqr(A, B, Q, R)

K = np.array(K)

def lqr_controller(q_err_vec, w):
    """
    LQR attitude controller
    """

    x = np.concatenate([q_err_vec, w])

    tau = -K @ x

    # Reaction wheel saturation
    tau = np.clip(tau, -0.25, 0.25)

    return tau