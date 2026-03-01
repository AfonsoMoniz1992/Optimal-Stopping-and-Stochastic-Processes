r"""Simulate the Vasicek interest rate model.

The Vasicek process \(r_t\) satisfies the stochastic differential equation

\[
    dr_t = \kappa(\theta - r_t)\, dt + \sigma\, dB_t,
\]

where
- \(\kappa > 0\) is the mean-reversion speed,
- \(\theta\) the long-run mean level,
- \(\sigma\) the instantaneous volatility,
- \(B_t\) a standard Brownian motion.

The process has a Gaussian (normal) stationary distribution with mean
\(\theta\) and variance \(\sigma^2 / (2\kappa)\), which means it can take
negative values — a known limitation compared with the CIR model.

The exact transition density is also Gaussian:

\[
    r_t \mid r_s \sim \mathcal{N}\!\left(
        \theta + (r_s - \theta)e^{-\kappa(t-s)},\;
        \frac{\sigma^2}{2\kappa}\bigl(1 - e^{-2\kappa(t-s)}\bigr)
    \right).
\]

This script simulates via the Euler–Maruyama scheme and overlays the
long-run mean and the \(\pm 1\sigma\) stationary band.

Seminal reference
-----------------
Vasicek, O. (1977). "An equilibrium characterization of the term structure."
Journal of Financial Economics, 5(2), 177–188.
https://doi.org/10.1016/0304-405X(77)90016-2
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Model parameters -------------------------------------------------------
kappa = 1.5   # mean-reversion speed
theta = 0.05  # long-run mean level
sigma = 0.02  # volatility
r0 = 0.02     # initial rate
T = 5.0       # final time
n = 1000      # number of steps

# --- Time grid --------------------------------------------------------------
t = np.linspace(0.0, T, n + 1)
dt = t[1] - t[0]

# --- Brownian increments ----------------------------------------------------
rng = np.random.default_rng(seed=42)
dB = rng.normal(0.0, np.sqrt(dt), size=n)

# --- Euler–Maruyama integration ---------------------------------------------
r = np.empty(n + 1)
r[0] = r0
for i in range(n):
    r[i + 1] = r[i] + kappa * (theta - r[i]) * dt + sigma * dB[i]

# --- Stationary standard deviation ------------------------------------------
sigma_stat = sigma / np.sqrt(2 * kappa)

# --- Plot -------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(t, r, label=r"$r_t$ (Vasicek)")
plt.axhline(theta, color="black", linestyle="--", linewidth=0.8,
            label=r"long-run mean $\theta$")
plt.axhline(theta + sigma_stat, color="grey", linestyle=":", linewidth=0.8,
            label=r"$\theta \pm \sigma_{\infty}$")
plt.axhline(theta - sigma_stat, color="grey", linestyle=":", linewidth=0.8)
plt.xlabel("Time")
plt.ylabel(r"$r(t)$")
plt.title("Vasicek interest rate model")
plt.legend()
plt.tight_layout()
plt.show()
