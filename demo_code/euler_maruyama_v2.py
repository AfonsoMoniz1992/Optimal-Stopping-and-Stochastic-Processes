"""Simulate an Ornstein–Uhlenbeck process with the Euler–Maruyama method.

The process satisfies the stochastic differential equation

    dX_t = theta * (mu - X_t) dt + sigma dB_t,

where ``theta`` is the mean‑reversion rate, ``mu`` the long‑run mean, and
``sigma`` the volatility.  This script generates a single sample path using a
simple Euler–Maruyama discretization.
"""

import numpy as np
import matplotlib.pyplot as plt


# --- Model parameters -----------------------------------------------------
t_0 = 0.0
t_end = 2.0
length = 1000
theta = 1.1
mu = 0.8
sigma = 0.3


# --- Time grid and initial condition --------------------------------------
t = np.linspace(t_0, t_end, length)             # time axis
dt = np.mean(np.diff(t))                        # fixed step size
y = np.zeros(length)
y[0] = np.random.normal(loc=0.0, scale=1.0)     # initial state


# --- Drift, diffusion, and noise ------------------------------------------
drift = lambda y, t: theta * (mu - y)           # mean-reverting drift
diffusion = lambda y, t: sigma                  # constant diffusion
# Pre-generate Brownian increments scaled by sqrt(dt)
noise = np.random.normal(loc=0.0, scale=1.0, size=length) * np.sqrt(dt)


# --- Euler–Maruyama integration ------------------------------------------
for i in range(1, length):
    y[i] = (
        y[i - 1]
        + drift(y[i - 1], i * dt) * dt
        + diffusion(y[i - 1], i * dt) * noise[i]
    )


# --- Plot the result ------------------------------------------------------
plt.plot(t, y)
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("Ornstein–Uhlenbeck sample path (Euler–Maruyama)")
plt.show()

