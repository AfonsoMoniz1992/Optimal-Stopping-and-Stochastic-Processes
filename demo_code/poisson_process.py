r"""Simulate a homogeneous Poisson process.

A Poisson process \(N_t\) with intensity \(\lambda > 0\) counts the number of
random events that have occurred by time \(t\).  It has independent increments
and

\[
    \mathbb{P}\big(N_{t+\Delta t} - N_t = k\big)
    = e^{-\lambda \Delta t}\frac{(\lambda \Delta t)^k}{k!}, \qquad k = 0,1,2,\ldots
\]

This script generates event times by drawing exponential inter-arrival times and
plots the resulting step function for the counting process.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Model parameters -------------------------------------------------------
lam = 3.0   # intensity \lambda
T = 5.0     # simulate on [0,T]

# --- Generate jump times ----------------------------------------------------
rng = np.random.default_rng(seed=1)
t = 0.0
jumps = []
while t < T:
    t += rng.exponential(1.0 / lam)
    if t < T:
        jumps.append(t)
arrival_times = np.array(jumps)

# --- Build counting process on a fine grid ---------------------------------
t_grid = np.linspace(0.0, T, 1000)
N_t = np.searchsorted(arrival_times, t_grid, side="right")

# --- Plot ------------------------------------------------------------------
plt.step(t_grid, N_t, where="post")
plt.xlabel("Time")
plt.ylabel("N(t)")
plt.title(f"Poisson process sample path (\u03bb={lam})")
plt.show()
