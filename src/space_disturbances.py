# src/space_disturbances.py

import numpy as np


# ==========================================================
# GRAVITY GRADIENT TORQUE
# ==========================================================

def gravity_gradient_torque(q, I):

    """
    Simplified gravity-gradient torque model
    """

    mu = 3.986e14          # Earth gravitational parameter
    r_orbit = 6771e3       # 400 km orbit radius

    # Body z-axis approximation
    z_body = np.array([0, 0, 1])

    factor = 3 * mu / (r_orbit**3)

    torque = factor * np.cross(
        z_body,
        I @ z_body
    )

    return torque


# ==========================================================
# AERODYNAMIC DRAG TORQUE
# ==========================================================

def aerodynamic_drag_torque(w):

    """
    Simplified aerodynamic drag torque
    """

    Cd = 2.2
    rho = 1e-12
    A = 0.01
    v = 7700

    coeff = 0.5 * rho * Cd * A * v**2

    torque = -coeff * 1e-3 * w

    return torque


# ==========================================================
# SOLAR RADIATION PRESSURE TORQUE
# ==========================================================

def solar_pressure_torque():

    """
    Simplified SRP torque model
    """

    P = 4.5e-6
    A = 0.02
    lever = 0.05

    torque_mag = P * A * lever

    torque = np.array([
        torque_mag,
        -0.5 * torque_mag,
        0.3 * torque_mag
    ])

    return torque


# ==========================================================
# TOTAL DISTURBANCE
# ==========================================================

def total_disturbance_torque(q, w, I):

    tau_gg = gravity_gradient_torque(q, I)

    tau_drag = aerodynamic_drag_torque(w)

    tau_srp = solar_pressure_torque()

    total = tau_gg + tau_drag + tau_srp

    return total