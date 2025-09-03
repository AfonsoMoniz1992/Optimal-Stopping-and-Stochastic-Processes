
r"""Simulate an Ornstein–Uhlenbeck process using the Euler–Maruyama scheme.

The Ornstein–Uhlenbeck SDE is

\[
    dX_t = \theta (\mu - X_t)\, dt + \sigma\, dB_t,
\]

whose Euler–Maruyama discretization for step size \(\Delta t\) reads

\[
    X_{n+1} = X_n + \theta(\mu - X_n)\Delta t + \sigma \Delta B_n.
\]

The script generates a single sample path.
"""
=======



import numpy as np
import matplotlib.pyplot as plt


# --- Model parameters -----------------------------------------------------
t_0 = 0.0       # initial time
t_end = 2.0     # final time
length = 1000   # number of time steps
theta = 1.1     # mean-reversion rate
mu = 0.8        # long-run mean
sigma = 0.3     # volatility

# --- Time grid and initial condition --------------------------------------
t = np.linspace(t_0, t_end, length)          # time axis
dt = np.mean(np.diff(t))                     # uniform step size
y = np.zeros(length)
y[0] = np.random.normal(loc=0.0, scale=1.0)  # initial state

# --- Drift, diffusion, and noise ------------------------------------------
drift = lambda y, t: theta * (mu - y)        # deterministic drift
diffusion = lambda y, t: sigma               # constant diffusion
noise = np.random.normal(                    # Brownian increments \Delta B
    loc=0.0, scale=np.sqrt(dt), size=length
)

=======

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

