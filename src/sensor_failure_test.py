import numpy as np
import matplotlib.pyplot as plt

from src.multiplicative_ekf import MultiplicativeEKF
from src.dynamics import step
from src.controllers import pd_controller
from src.sensors import measure_gyro, body_vector_from_inertial
from src.quaternion_utils import quat_mult, quat_conj

print("=== Sensor Failure Recovery Test ===")

I = np.diag([0.02, 0.025, 0.03])

dt = 0.02
t_final = 60

tvec = np.arange(0, t_final, dt)

q_des = np.array([1., 0., 0., 0.])

axis = np.array([0.3, -0.4, 0.2])
axis /= np.linalg.norm(axis)

angle = 0.2

q0 = np.concatenate(([np.cos(angle/2)],
                     np.sin(angle/2)*axis))

w0 = np.array([0.02, -0.01, 0.015])

state = np.concatenate([q0, w0])

ekf = MultiplicativeEKF(dt)

v_inertial = np.array([1., 0.1, -0.05])
v_inertial /= np.linalg.norm(v_inertial)

bias_true = np.array([1e-4, 0., -5e-5])

attitude_error = []

for t in tvec:

    q_true = state[:4]
    w_true = state[4:]

    gyro = measure_gyro(
        w_true,
        bias_true,
        5e-4
    )

    star_tracker_available = not (20 <= t <= 30)

    v_body = body_vector_from_inertial(
        q_true,
        v_inertial,
        1e-3,
        sensor_available=star_tracker_available
    )

    ekf.predict(gyro)

    if v_body is not None:
        ekf.update_vector(v_body, v_inertial)

    q_est, b_est, _ = ekf.get_state()

    w_est = gyro - b_est

    tau = pd_controller(
        q_des,
        q_est,
        w_est
    )

    state = step(
        state,
        tau,
        np.zeros(3),
        I,
        dt
    )

    q_err = quat_mult(
        q_des,
        quat_conj(state[:4])
    )

    err_deg = 2*np.degrees(
        np.arccos(
            np.clip(q_err[0], -1, 1)
        )
    )

    attitude_error.append(err_deg)

plt.figure(figsize=(10,5))
plt.plot(tvec, attitude_error)

plt.axvspan(
    20,
    30,
    alpha=0.3,
    label="Star Tracker Failure"
)

plt.xlabel("Time (s)")
plt.ylabel("Attitude Error (deg)")
plt.title("Attitude Recovery During Sensor Outage")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/sensor_failure_recovery.png",
    dpi=300
)

plt.show()

print(
    "✅ Saved: results/sensor_failure_recovery.png"
)