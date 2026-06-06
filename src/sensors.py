# src/sensors.py
import numpy as np
from src.quaternion_utils import quat_mult, quat_conj

def measure_gyro(true_w, bias, sigma):
    """
    Simulate a noisy gyroscope measurement.
    true_w: true angular rate [wx, wy, wz]
    bias: constant gyro bias
    sigma: standard deviation of noise
    """
    noise = np.random.randn(3) * sigma
    return true_w + bias + noise

def body_vector_from_inertial(
        q,
        v_inertial,
        sigma,
        sensor_available=True,
        dropout_probability=0.0):
    """
    Simulate body-frame measurement of inertial vector.

    New features:
    - Sensor dropout
    - Star tracker blinding
    - Measurement noise
    """

    # Random dropout
    if np.random.rand() < dropout_probability:
        return None

    # Sensor unavailable (blinding)
    if not sensor_available:
        return None

    v_body = quat_mult(
        quat_mult(
            q,
            np.concatenate(([0.0], v_inertial))
        ),
        quat_conj(q)
    )[1:]

    noise = np.random.randn(3) * sigma

    return v_body + noise