# src/compare_lqr_vs_ai.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.multiplicative_ekf import MultiplicativeEKF
from src.dynamics import step
from src.sensors import measure_gyro, body_vector_from_inertial
from src.controllers import pd_controller
from src.ai_controller import AIController
from src.lqr_controller import lqr_controller
from src.quaternion_utils import quat_mult, quat_conj
from src.space_disturbances import total_disturbance_torque


# =========================================================
# SIMULATION FUNCTION
# =========================================================

def run_sim(controller_type="PD"):

    # Spacecraft inertia matrix
    I = np.diag([0.02, 0.025, 0.03])

    # Simulation settings
    dt = 0.02
    t_final = 40.0
    tvec = np.arange(0, t_final, dt)

    # Desired quaternion
    q_des = np.array([1., 0., 0., 0.])

    # Initial orientation
    axis = np.array([0.3, -0.4, 0.2])
    axis /= np.linalg.norm(axis)

    angle = 0.25

    q0 = np.concatenate((
        [np.cos(angle / 2)],
        np.sin(angle / 2) * axis
    ))

    # Initial angular velocity
    w0 = np.array([0.03, -0.02, 0.015])

    # Initial state
    state = np.concatenate([q0, w0])

    # EKF estimator
    ekf = MultiplicativeEKF(dt)

    # AI model
    ai = AIController("results/ai_model.pt")

    # Inertial reference vector
    v_inertial = np.array([1., 0.1, -0.05])
    v_inertial /= np.linalg.norm(v_inertial)

    # True gyro bias
    bias_true = np.array([1e-4, 0., -5e-5])

    # Logs
    err_log = []
    tau_log = []

    # =====================================================
    # MAIN SIMULATION LOOP
    # =====================================================

    for t in tvec:

        q_true = state[:4]
        w_true = state[4:]

        # -------------------------------------------------
        # SENSOR MEASUREMENTS
        # -------------------------------------------------

        gyro = measure_gyro(
            w_true,
            bias_true,
            5e-4
        )

        v_body = body_vector_from_inertial(
            q_true,
            v_inertial,
            1e-3
        )

        # -------------------------------------------------
        # EKF ESTIMATION
        # -------------------------------------------------

        ekf.predict(gyro)

        ekf.update_vector(
            v_body,
            v_inertial
        )

        q_est, b_est, _ = ekf.get_state()

        w_est = gyro - b_est

        # Quaternion error vector
        q_err_vec = quat_mult(
            q_est,
            quat_conj(q_des)
        )[1:4]

        # =================================================
        # CONTROLLER SELECTION
        # =================================================

        # -----------------------------
        # PD Controller
        # -----------------------------
        if controller_type == "PD":

            tau = pd_controller(
                q_des,
                q_est,
                w_est
            )

        # -----------------------------
        # LQR Controller
        # -----------------------------
        elif controller_type == "LQR":

            tau = lqr_controller(
                q_err_vec,
                w_est
            )

        # -----------------------------
        # HYBRID AI + LQR
        # -----------------------------
        elif controller_type == "HYBRID":

            # Stable LQR backbone
            tau_lqr = lqr_controller(
                q_err_vec,
                w_est
            )

            # AI compensation torque
            tau_ai = ai.compute(
                q_err_vec,
                w_est
            )

            # Small AI blending factor
            alpha = 0.05

            # Hybrid torque
            tau = tau_lqr + alpha * tau_ai

            # Final actuator saturation
            tau = np.clip(
                tau,
                -0.25,
                0.25
            )

        else:
            raise ValueError("Unknown controller type")

        # =================================================
        # REAL ORBITAL DISTURBANCES
        # =================================================

        disturbance = total_disturbance_torque(
            q_true,
            w_true,
            I
        )

        # Additional impulse disturbance
        if 12.0 <= t < 12.5:

            disturbance += np.array([
                2e-3,
                -1.5e-3,
                1e-3
            ])

           
        # =================================================
        # PROPAGATE SPACECRAFT DYNAMICS
        # =================================================

        state = step(
            state,
            tau,
            disturbance,
            I,
            dt
        )

        # =================================================
        # TRUE ATTITUDE ERROR
        # =================================================

        q_err_full = quat_mult(
            q_des,
            quat_conj(state[:4])
        )

        ang_err = 2 * np.degrees(
            np.arccos(
                np.clip(q_err_full[0], -1, 1)
            )
        )

        # Save logs
        err_log.append(ang_err)

        tau_log.append(
            np.linalg.norm(tau)
        )

    return (
        tvec,
        np.array(err_log),
        np.array(tau_log)
    )


# =========================================================
# RUN ALL CONTROLLERS
# =========================================================

print("\n=== Running Controller Comparison ===\n")

# PD
t, err_pd, tau_pd = run_sim("PD")
print("✅ PD complete")

# LQR
t, err_lqr, tau_lqr = run_sim("LQR")
print("✅ LQR complete")

# HYBRID
t, err_hybrid, tau_hybrid = run_sim("HYBRID")
print("✅ HYBRID complete")


# =========================================================
# PERFORMANCE METRICS
# =========================================================

def rms(x):
    return np.sqrt(np.mean(x**2))

print("\n===== PERFORMANCE SUMMARY =====\n")

print(f"PD RMS Error      : {rms(err_pd):.3f} deg")
print(f"LQR RMS Error     : {rms(err_lqr):.3f} deg")
print(f"Hybrid RMS Error  : {rms(err_hybrid):.3f} deg")


# =========================================================
# SAVE RESULTS
# =========================================================

os.makedirs("results", exist_ok=True)

df = pd.DataFrame({
    "time": t,

    "pd_error": err_pd,
    "lqr_error": err_lqr,
    "hybrid_error": err_hybrid,

    "pd_tau": tau_pd,
    "lqr_tau": tau_lqr,
    "hybrid_tau": tau_hybrid
})

df.to_csv(
    "results/lqr_vs_ai_results.csv",
    index=False
)

print("\n✅ Results saved to results/lqr_vs_ai_results.csv")


# =========================================================
# ERROR COMPARISON PLOT
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(t, err_pd, label="PD")
plt.plot(t, err_lqr, label="LQR")
plt.plot(t, err_hybrid, label="Hybrid AI")

plt.xlabel("Time (s)")
plt.ylabel("Pointing Error (deg)")
plt.title("Controller Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/lqr_vs_ai_error.png",
    dpi=300
)

plt.show()


# =========================================================
# TORQUE COMPARISON PLOT
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(t, tau_pd, label="PD")
plt.plot(t, tau_lqr, label="LQR")
plt.plot(t, tau_hybrid, label="Hybrid AI")

plt.xlabel("Time (s)")
plt.ylabel("Torque Norm")
plt.title("Torque Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/lqr_vs_ai_torque.png",
    dpi=300
)

plt.show()

print("\n✅ Comparison plots saved.")