import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.dynamics import step
from src.sensors import measure_gyro, body_vector_from_inertial
from src.multiplicative_ekf import MultiplicativeEKF
from src.controllers import pd_controller
from src.quaternion_utils import quat_mult, quat_conj
from src.ai_controller import AIController

print("=== Multi-Orbit Mission Simulation ===")

I = np.diag([0.02, 0.025, 0.03])

dt = 0.1

orbit_period = 5400.0      # ~90 min LEO orbit
num_orbits = 3

t_final = orbit_period * num_orbits

tvec = np.arange(0, t_final, dt)

axis = np.array([0.3, -0.4, 0.2])
axis /= np.linalg.norm(axis)

angle = 1.0

q0 = np.concatenate(([np.cos(angle/2)],
                     np.sin(angle/2)*axis))

state = np.concatenate([
    q0,
    np.array([0.02, -0.01, 0.015])
])

ekf = MultiplicativeEKF(dt)

v_inertial = np.array([1.0, 0.1, -0.05])
v_inertial /= np.linalg.norm(v_inertial)

bias_true = np.array([1e-4, 0.0, -5e-5])

gyro_sigma = 5e-4
vec_sigma = 1e-3

q_des = np.array([1.,0.,0.,0.])

ai = AIController(
    model_path="results/ai_model.pt"
)

error_log = []
torque_log = []

for t in tvec:

    q_true = state[:4]
    w_true = state[4:]

    gyro = measure_gyro(
        w_true,
        bias_true,
        gyro_sigma
    )

    v_body = body_vector_from_inertial(
        q_true,
        v_inertial,
        vec_sigma
    )

    ekf.predict(gyro)

    if v_body is not None:
        ekf.update_vector(
            v_body,
            v_inertial
        )

    q_est, bias_est, _ = ekf.get_state()

    w_est = gyro - bias_est

    tau_pd = pd_controller(
        q_des,
        q_est,
        w_est
    )

    q_err = quat_mult(
        q_est,
        quat_conj(q_des)
    )[1:]

    tau_ai = np.zeros(3)

    tau = tau_pd

    disturbance = np.array([
        0.001*np.sin(2*np.pi*t/orbit_period),
        0.0007*np.cos(2*np.pi*t/orbit_period),
        0.0005*np.sin(4*np.pi*t/orbit_period)
    ])

    state = step(
        state,
        tau,
        disturbance,
        I,
        dt
    )

    err_q = quat_mult(
        q_des,
        quat_conj(state[:4])
    )

    err_deg = 2*np.degrees(
        np.arccos(
            np.clip(err_q[0], -1, 1)
        )
    )

    error_log.append(err_deg)
    torque_log.append(np.linalg.norm(tau))

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10,4))
plt.plot(
    tvec/3600,
    error_log
)
plt.xlabel("Mission Time (hours)")
plt.ylabel("Pointing Error (deg)")
plt.title("Multi-Orbit Pointing Performance")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/multi_orbit_error.png",
    dpi=300
)

plt.figure(figsize=(10,4))
plt.plot(
    tvec/3600,
    torque_log
)
plt.xlabel("Mission Time (hours)")
plt.ylabel("Torque Norm (Nm)")
plt.title("Multi-Orbit Control Effort")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/multi_orbit_torque.png",
    dpi=300
)

df = pd.DataFrame({
    "time_sec": tvec,
    "pointing_error_deg": error_log,
    "torque_norm": torque_log
})

df.to_csv(
    "results/multi_orbit_results.csv",
    index=False
)

print("\n===== MISSION SUMMARY =====")
print(
    f"Mean Pointing Error : {np.mean(error_log):.3f} deg"
)
print(
    f"Max Pointing Error  : {np.max(error_log):.3f} deg"
)
print(
    f"Mean Torque         : {np.mean(torque_log):.6f} Nm"
)

print("\n✅ results/multi_orbit_error.png")
print("✅ results/multi_orbit_torque.png")
print("✅ results/multi_orbit_results.csv")

plt.show()