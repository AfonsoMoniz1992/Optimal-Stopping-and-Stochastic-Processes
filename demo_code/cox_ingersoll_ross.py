r"""Simulate the Cox–Ingersoll–Ross (CIR) interest rate model.

The CIR process \(R_t\) satisfies the stochastic differential equation

\[
    dR_t = \kappa(\theta - R_t)\, dt + \sigma \sqrt{R_t}\, dB_t,
\]

where \(\kappa\) is the mean-reversion speed, \(\theta\) the long-run level,
\(\sigma\) the volatility, and \(B_t\) a standard Brownian motion.  A simple
Euler–Maruyama discretization with step size \(\Delta t\) is

\[
    R_{n+1} = R_n + \kappa(\theta - R_n)\Delta t + \sigma\sqrt{R_n}\, \Delta B_n.
\]

This implementation uses the approximation above and clips negative values that
may arise from discretization error.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Model parameters -------------------------------------------------------
kappa = 2.0   # mean-reversion speed
theta = 0.05  # long-run mean level
sigma = 0.1   # volatility
R0 = 0.03     # initial rate
T = 5.0       # final time
n = 1000      # number of steps

# --- Time grid and initial state -------------------------------------------
t = np.linspace(0.0, T, n + 1)
dt = t[1] - t[0]
R = np.empty(n + 1)
R[0] = R0

# --- Brownian increments ----------------------------------------------------
rng = np.random.default_rng(seed=1)
dB = rng.normal(0.0, np.sqrt(dt), size=n)

# --- Euler–Maruyama integration --------------------------------------------
for i in range(n):
    drift = kappa * (theta - R[i]) * dt
    diffusion = sigma * np.sqrt(max(R[i], 0.0)) * dB[i]
    R[i + 1] = R[i] + drift + diffusion
    R[i + 1] = max(R[i + 1], 0.0)  # enforce non-negativity

# --- Plot ------------------------------------------------------------------
plt.plot(t, R)
plt.xlabel("Time")
plt.ylabel("R(t)")
plt.title("Cox–Ingersoll–Ross interest rate sample path")
plt.show()
