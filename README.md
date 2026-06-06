# \# Hybrid AI-Powered Satellite Attitude Determination and Control System (ADCS)

# 

# !\[Python](https://img.shields.io/badge/Python-3.x-blue)

# !\[PyTorch](https://img.shields.io/badge/PyTorch-Neural\_Network-orange)

# !\[Spacecraft GNC](https://img.shields.io/badge/Spacecraft-GNC-green)

# !\[MEKF](https://img.shields.io/badge/MEKF-Attitude\_Estimation-red)

# !\[ADCS](https://img.shields.io/badge/Satellite-ADCS-blueviolet)

# 

# \---

# 

# \## Overview

# 

# This project presents a Hybrid Artificial Intelligence Powered Satellite Attitude Determination and Control System (ADCS) developed for spacecraft attitude estimation, stabilization, and control under realistic orbital operating conditions.

# 

# The framework integrates quaternion-based spacecraft dynamics, Multiplicative Extended Kalman Filtering (MEKF), classical control techniques, and neural-network-based control augmentation into a unified simulation environment.

# 

# The objective is to investigate how Artificial Intelligence can be combined with traditional spacecraft control systems to improve attitude control performance, disturbance rejection capability, and mission robustness.

# 

# \---

# 

# \## Project Scope

# 

# This project develops a complete spacecraft Attitude Determination and Control System (ADCS) integrating state estimation, classical control, artificial intelligence, sensor modeling, disturbance simulation, and mission-level validation within a unified spacecraft Guidance, Navigation and Control (GNC) framework.

# 

# \---

# 

# \## Project Highlights

# 

# \- Developed a complete spacecraft Guidance, Navigation and Control (GNC) framework.

# \- Implemented Multiplicative Extended Kalman Filtering (MEKF) for spacecraft attitude estimation.

# \- Designed a Hybrid AI-Augmented Attitude Control architecture.

# \- Modeled realistic spacecraft sensors including gyroscopes and reference vector sensors.

# \- Simulated disturbance torques and reaction wheel actuation.

# \- Evaluated robustness using Monte Carlo simulations and sensor failure recovery tests.

# \- Performed multi-orbit mission simulations for long-duration validation.

# \- Compared PD, LQR, and AI-based spacecraft controllers.

# 

# \---

# 

# \## System Architecture

# 

# !\[Architecture](docs/images/architecture\_diagram.png)

# 

# \---

# 

# \## Key Features

# 

# \- Quaternion-Based Attitude Representation

# \- Three-Axis Spacecraft Rotational Dynamics

# \- Multiplicative Extended Kalman Filter (MEKF)

# \- Gyroscope and Reference Vector Sensor Models

# \- Sensor Noise and Bias Modeling

# \- PD Controller

# \- Linear Quadratic Regulator (LQR)

# \- AI Neural Network Controller

# \- Hybrid AI-Augmented Attitude Control

# \- Reaction Wheel Actuation

# \- Disturbance Torque Modeling

# \- Monte Carlo Robustness Analysis

# \- Sensor Failure Recovery Simulation

# \- Multi-Orbit Mission Validation

# 

# \---

# 

# \## Mathematical Framework

# 

# \### Quaternion Kinematics

# 

# q̇ = ½ Ω(ω) q

# 

# \### Spacecraft Rotational Dynamics

# 

# Iω̇ + ω × (Iω) = τ + τd

# 

# \### Hybrid Control Law

# 

# τ = τPD + τAI

# 

# where:

# 

# \- τPD = Classical PD control torque

# \- τAI = Neural network compensation torque

# 

# \---

# 

# \## Validation Framework

# 

# \### Controller Comparison

# 

# Comparison between:

# 

# \- PD Controller

# \- LQR Controller

# \- Hybrid AI Controller

# 

# \### Monte Carlo Analysis

# 

# Repeated simulations under varying:

# 

# \- Sensor Noise

# \- Disturbance Torques

# \- Initial Conditions

# 

# \### Sensor Failure Recovery

# 

# Simulation of temporary Star Tracker outages to evaluate estimator robustness and recovery performance.

# 

# \### Multi-Orbit Mission Simulation

# 

# Long-duration orbital simulations used to evaluate:

# 

# \- Pointing Stability

# \- Control Effort

# \- Mission-Level Robustness

# 

# \---

# 

# \# Results

# 

# \## Controller Performance Comparison

# 

# The controller comparison study evaluates pointing performance achieved by classical and AI-assisted control strategies under identical spacecraft operating conditions.

# 

# | Controller | RMS Pointing Error (deg) |

# |------------|-------------------------|

# | PD Controller | 8.161 |

# | LQR Controller | 6.498 |

# | Hybrid AI Controller | 10.130 |

# 

# \### Controller Comparison

# 

# !\[Controller Comparison](results/lqr\_vs\_ai\_error.png)

# 

# \---

# 

# \## Attitude Tracking Performance

# 

# !\[Attitude Error](results/attitude\_error\_plot.png)

# 

# \---

# 

# \## Sensor Failure Recovery

# 

# The ADCS successfully maintained attitude estimation during temporary sensor outages and recovered after sensor restoration.

# 

# !\[Sensor Failure Recovery](results/sensor\_failure\_recovery.png)

# 

# \---

# 

# \## Multi-Orbit Mission Performance

# 

# | Metric | Value |

# |----------|----------|

# | Mean Pointing Error | 48.528 deg |

# | Maximum Pointing Error | 139.733 deg |

# 

# \### Multi-Orbit Pointing Error

# 

# !\[Multi-Orbit Error](results/multi\_orbit\_error.png)

# 

# \### Multi-Orbit Control Torque

# 

# !\[Multi-Orbit Torque](results/multi\_orbit\_torque.png)

# 

# \---

# 

# \## Discussion

# 

# The Hybrid AI Controller serves as a proof-of-concept AI augmentation framework for spacecraft attitude control.

# 

# While the current implementation does not yet outperform the optimally tuned LQR controller, the results demonstrate successful integration of machine learning techniques within a spacecraft Guidance, Navigation and Control (GNC) architecture.

# 

# The project establishes a foundation for future AI-enhanced spacecraft control research involving adaptive control, reinforcement learning, and autonomous on-orbit decision-making.

# 

# \---

# 

# \## Monte Carlo Validation

# 

# The Monte Carlo validation framework evaluates robustness under varying sensor noise levels and disturbance conditions.

# 

# Outputs include:

# 

# \- Pointing Error Statistics

# \- Torque Usage Statistics

# \- Estimation Performance Metrics

# \- Controller Robustness Assessment

# 

# \---

# 

# \## Repository Structure

# 

# ```text

# hybrid\_adcs/

# │

# ├── src/

# │   ├── sim.py

# │   ├── dynamics.py

# │   ├── estimator.py

# │   ├── multiplicative\_ekf.py

# │   ├── sensors.py

# │   ├── controllers.py

# │   ├── lqr\_controller.py

# │   ├── ai\_controller.py

# │   ├── train\_ai.py

# │   ├── compare\_lqr\_vs\_ai.py

# │   ├── sensor\_failure\_test.py

# │   ├── multi\_orbit\_sim.py

# │   ├── monte\_carlo\_ai.py

# │   └── validate.py

# │

# ├── docs/

# │   └── images/

# │       └── architecture\_diagram.png

# │

# ├── results/

# │   ├── lqr\_vs\_ai\_error.png

# │   ├── sensor\_failure\_recovery.png

# │   ├── attitude\_error\_plot.png

# │   ├── multi\_orbit\_error.png

# │   ├── multi\_orbit\_torque.png

# │   └── \*.csv

# │

# ├── report/

# │   └── Hybrid\_ADCS\_Report.pdf

# │

# ├── requirements.txt

# │

# └── README.md

# ```

# 

# \---

# 

# \## Installation

# 

# Clone the repository:

# 

# ```bash

# git clone https://github.com/yourusername/hybrid\_adcs.git

# cd hybrid\_adcs

# ```

# 

# Install dependencies:

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \---

# 

# \## Requirements

# 

# ```text

# numpy

# scipy

# matplotlib

# pandas

# torch

# tqdm

# control

# ```

# 

# \---

# 

# \## Running Simulations

# 

# \### Baseline ADCS Simulation

# 

# ```bash

# python src/sim.py

# ```

# 

# \### Train Neural Controller

# 

# ```bash

# python src/train\_ai.py

# ```

# 

# \### Compare LQR and AI Controller

# 

# ```bash

# python src/compare\_lqr\_vs\_ai.py

# ```

# 

# \### Sensor Failure Recovery Test

# 

# ```bash

# python src/sensor\_failure\_test.py

# ```

# 

# \### Multi-Orbit Mission Simulation

# 

# ```bash

# python src/multi\_orbit\_sim.py

# ```

# 

# \### Monte Carlo Validation

# 

# ```bash

# python src/monte\_carlo\_ai.py

# ```

# 

# \---

# 

# \## Technologies Used

# 

# \- Python

# \- NumPy

# \- SciPy

# \- Pandas

# \- Matplotlib

# \- PyTorch

# \- Multiplicative Extended Kalman Filter (MEKF)

# \- Quaternion-Based Attitude Representation

# \- Proportional-Derivative (PD) Control

# \- Linear Quadratic Regulator (LQR)

# \- Neural Network-Based Attitude Control

# \- Monte Carlo Simulation

# 

# \---

# 

# \## Applications

# 

# \- Satellite Attitude Determination and Control

# \- CubeSat ADCS Development

# \- Spacecraft Guidance, Navigation and Control (GNC)

# \- Autonomous Spacecraft Operations

# \- AI for Aerospace Systems

# \- Fault-Tolerant Spacecraft Control

# \- Spacecraft Mission Analysis

# 

# \---

# 

# \## Future Improvements

# 

# \- Deep Reinforcement Learning-Based Attitude Control

# \- Adaptive Gain Scheduling

# \- Reaction Wheel Failure Recovery

# \- CubeSat Hardware-in-the-Loop Validation

# \- Star Tracker Image Simulation

# \- Disturbance Observer Integration

# \- Autonomous On-Orbit Reconfiguration

# \- Real-Time Embedded Deployment

# 

# \---

# 

# \## Full Technical Report

# 

# A detailed 61-page project report including:

# 

# \- System Architecture

# \- Mathematical Modeling

# \- Spacecraft Dynamics

# \- Multiplicative Extended Kalman Filter (MEKF)

# \- AI-Augmented Control

# \- Validation Framework

# \- Monte Carlo Analysis

# \- Sensor Failure Recovery

# \- Multi-Orbit Mission Simulation

# \- Results and Discussion

# 

# is available in:

# 

# 📄 `report/Hybrid\_ADCS\_Report.pdf`

# 

# \---

# 

# \## Author

# 

# \*\*Amaraneni Vinitha\*\*

# 

# B.Tech Aeronautical Engineering

# 

# Aircraft Systems • Flight Dynamics • Control Systems • Spacecraft Dynamics • Artificial Intelligence for Aerospace

