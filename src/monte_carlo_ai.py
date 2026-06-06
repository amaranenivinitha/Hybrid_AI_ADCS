# src/monte_carlo_ai.py
import numpy as np
import os
import pandas as pd

from src.multiplicative_ekf import MultiplicativeEKF
from src.dynamics import step
from src.sensors import measure_gyro, body_vector_from_inertial
from src.ai_controller import AIController
from src.controllers import pd_controller
from src.quaternion_utils import quat_mult, quat_conj


def run_single_trial(use_ai: bool,
                     seed: int = 0,
                     noise_scale: float = 1.0,
                     disturb_scale: float = 1.0):
    """
    One simulation run with given noise/disturbance scaling.
    Returns: rms_deg, max_deg, avg_torque
    """
    np.random.seed(seed)

    # Inertia and time settings
    I = np.diag([0.02, 0.025, 0.03])
    dt = 0.02
    t_final = 40.0
    tvec = np.arange(0.0, t_final, dt)

    # Desired attitude
    q_des = np.array([1.0, 0.0, 0.0, 0.0])

    # Randomize initial attitude a bit
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0.1, 0.5)  # rad
    q0 = np.concatenate(([np.cos(angle / 2.0)],
                         np.sin(angle / 2.0) * axis))

    # Randomize initial angular velocity
    w0 = np.array([
        np.random.uniform(-0.03, 0.03),
        np.random.uniform(-0.03, 0.03),
        np.random.uniform(-0.03, 0.03),
    ])

    state = np.concatenate([q0, w0])

    # Estimator
    ekf = MultiplicativeEKF(dt)

    # AI controller (if used)
    ai = AIController(model_path="results/ai_model.pt") if use_ai else None

    # Reference inertial vector
    v_inertial = np.array([1.0, 0.1, -0.05])
    v_inertial /= np.linalg.norm(v_inertial)

    # True gyro bias (fixed)
    bias_true = np.array([1e-4, 0.0, -5e-5])

    # Base sensor noise
    gyro_sigma_base = 5e-4
    vec_sigma_base = 1e-3
    gyro_sigma = gyro_sigma_base * noise_scale
    vec_sigma = vec_sigma_base * noise_scale

    err_log = []
    tau_log = []

    for t in tvec:
        q_true = state[:4]
        w_true = state[4:]

        # Sensor measurements with scaled noise
        gyro = measure_gyro(w_true, bias_true, gyro_sigma)
        v_body = body_vector_from_inertial(q_true, v_inertial, vec_sigma)

        # EKF update
        ekf.predict(gyro)
        ekf.update_vector(v_body, v_inertial)
        q_est, b_est, _ = ekf.get_state()
        w_est = gyro - b_est

        # Attitude error (imaginary part)
        q_err_vec = quat_mult(q_est, quat_conj(q_des))[1:4]

        # PD control
        tau_pd = pd_controller(q_des, q_est, w_est)

        # AI compensation
        if use_ai and ai is not None:
            tau_ai = ai.compute(q_err_vec, w_est)
        else:
            tau_ai = np.zeros(3)

        tau = tau_pd + tau_ai

        # Disturbance impulse around t = 12s (scaled)
        if 12.0 <= t < 12.4:
            base_dist = np.array([5e-3, -4e-3, 3e-3])
            disturbance = disturb_scale * base_dist
        else:
            disturbance = np.zeros(3)

        # Propagate dynamics
        state = step(state, tau, disturbance, I, dt)

        # True pointing error (deg)
        q_err_full = quat_mult(q_des, quat_conj(state[:4]))
        ang_err_deg = 2.0 * np.degrees(
            np.arccos(np.clip(q_err_full[0], -1.0, 1.0))
        )

        err_log.append(ang_err_deg)
        tau_log.append(np.linalg.norm(tau))

    err_log = np.array(err_log)
    tau_log = np.array(tau_log)

    rms_deg = np.sqrt(np.mean(err_log**2))
    max_deg = np.max(err_log)
    avg_torque = np.mean(tau_log)

    return rms_deg, max_deg, avg_torque


def run_monte_carlo():
    """
    Runs multiple trials for PD and AI under
    different noise and disturbance levels.
    Saves results to results/mc_results.csv and prints summary.
    """
    os.makedirs("results", exist_ok=True)

    # Scenarios: (noise_scale, disturb_scale)
    scenarios = [
        (1.0, 1.0),
        (2.0, 1.0),
        (1.0, 1.5),
        (2.0, 1.5),
    ]
    scenario_names = [
        "nominal_noise_nominal_disturb",
        "high_noise_nominal_disturb",
        "nominal_noise_high_disturb",
        "high_noise_high_disturb",
    ]

    n_runs = 20  # per scenario per controller

    rows = []

    print("=== Monte Carlo Robustness Study (PD vs Hybrid AI) ===")
    for (noise_scale, disturb_scale), sname in zip(scenarios, scenario_names):
        print(f"\nScenario: {sname} (noise_scale={noise_scale}, "
              f"disturb_scale={disturb_scale})")

        rms_pd_list = []
        rms_ai_list = []
        for k in range(n_runs):
            seed_pd = 1000 + k
            seed_ai = 2000 + k

            # PD-only
            rms_pd, max_pd, avg_tau_pd = run_single_trial(
                use_ai=False,
                seed=seed_pd,
                noise_scale=noise_scale,
                disturb_scale=disturb_scale,
            )
            rows.append({
                "scenario": sname,
                "controller": "PD",
                "run": k,
                "rms_deg": rms_pd,
                "max_deg": max_pd,
                "avg_torque": avg_tau_pd,
            })
            rms_pd_list.append(rms_pd)

            # Hybrid AI
            rms_ai, max_ai, avg_tau_ai = run_single_trial(
                use_ai=True,
                seed=seed_ai,
                noise_scale=noise_scale,
                disturb_scale=disturb_scale,
            )
            rows.append({
                "scenario": sname,
                "controller": "AI",
                "run": k,
                "rms_deg": rms_ai,
                "max_deg": max_ai,
                "avg_torque": avg_tau_ai,
            })
            rms_ai_list.append(rms_ai)

        rms_pd_arr = np.array(rms_pd_list)
        rms_ai_arr = np.array(rms_ai_list)

        mean_pd = np.mean(rms_pd_arr)
        mean_ai = np.mean(rms_ai_arr)
        std_pd = np.std(rms_pd_arr)
        std_ai = np.std(rms_ai_arr)

        improvement = (1.0 - mean_ai / mean_pd) * 100.0

        print(f"  PD   : mean RMS = {mean_pd:.3f}°, std = {std_pd:.3f}°")
        print(f"  AI   : mean RMS = {mean_ai:.3f}°, std = {std_ai:.3f}°")
        print(f"  ==>  Avg Improvement: {improvement:.1f}%")

    # Save all runs to CSV
    df = pd.DataFrame(rows)
    out_path = os.path.join("results", "mc_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Monte Carlo results saved to {out_path}")


if __name__ == "__main__":
    run_monte_carlo()
