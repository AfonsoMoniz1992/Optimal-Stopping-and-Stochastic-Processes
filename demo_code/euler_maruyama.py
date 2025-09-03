r"""
Euler–Maruyama approximation for geometric Brownian motion.

We consider the stochastic differential equation

\[
    dX_t = r_0 X_t\,dt + \sigma_0 X_t\,dB_t,
\]

whose exact solution is

\[
    X_t = X_0 \exp\left((r_0 - \tfrac{1}{2}\sigma_0^2)t + \sigma_0 B_t\right).
\]

The Euler–Maruyama discretization for step size \(\Delta t\) is

\[
    Y_{n+1} = Y_n + r_0 Y_n \Delta t + \sigma_0 Y_n \Delta B_n.
\]

This script simulates both paths for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

# Number of simulated paths
num_sims = 1

# Number of time steps
N = 1500

# Initial value for both processes
X_0 = 1.0
Y_0 = X_0

# Time interval [t_0, T]
t_0 = 0.0
T = 10.0

# Parameters of the SDE
r_0 = 0.0       # drift coefficient
sigma_0 = 1.0   # diffusion coefficient

# Time discretization
dt = float(T - t_0) / N
t = np.arange(t_0, T, dt)

# Preallocate arrays for Brownian increments and paths
dB = np.zeros(N)           # \Delta B_n
B = np.zeros(N)            # B_n
X = np.zeros(N)            # exact solution
Y = np.zeros(N)            # Euler–Maruyama approximation

# Set initial conditions
dB[0] = 0.0
B[0] = 0.0
X[0] = X_0
Y[0] = Y_0

# Sample means across simulations (useful for many paths)
SX = np.zeros(N)
SY = np.zeros(N)

# Simulate
for n in range(num_sims):
    for i in range(1, t.size):
        # Brownian increment \Delta B_n ~ N(0, dt)
        dB[i] = np.random.normal(loc=0.0, scale=np.sqrt(dt))

        # Sample of Brownian motion B_t ~ N(0, t)
        B[i] = np.random.normal(loc=0.0, scale=np.sqrt(t[i]))

        # Exact solution at time step i
        X[i] = X_0 * np.exp((r_0 - 0.5 * sigma_0**2) * (i * dt) + sigma_0 * B[i])
        SX[i] += X[i] / num_sims

        # Euler–Maruyama step
        Y[i] = Y[i - 1] + r_0 * Y[i - 1] * dt + sigma_0 * Y[i - 1] * dB[i]
        SY[i] += Y[i] / num_sims

# Plot the results
plt.plot(t, SX, label="Exact solution")
plt.plot(t, SY, label="Euler–Maruyama")
plt.xlabel("Time")
plt.legend()
plt.show()
