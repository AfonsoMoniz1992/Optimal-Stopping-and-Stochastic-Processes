r"""Simulate a jump-to-default asset price process.

A jump-to-default (JTD) process models a risky asset whose price follows
geometric Brownian motion until a random default time, after which the price
jumps to zero.  One reduced-form representation is

\[
    dS_t = \mu S_t\, dt + \sigma S_t\, dB_t - S_{t-}\, dN_t,
\]

where
- \(\mu\) is the drift,
- \(\sigma\) the volatility,
- \(B_t\) standard Brownian motion,
- \(N_t\) a Poisson process with intensity \(\lambda\) that triggers default.

When \(N_t\) jumps, the asset is worthless and remains at zero thereafter.
This script simulates such a path and marks the default time.

Seminal references
------------------
Merton, R. C. (1974). "On the Pricing of Corporate Debt: The Risk Structure
of Interest Rates." Journal of Finance, 29(2), 449–470.
https://doi.org/10.1111/j.1540-6261.1974.tb03058.x
(Structural default model; GBM with an absorbing barrier.)

Jarrow, R. A. & Turnbull, S. M. (1995). "Pricing Derivatives on Financial
Securities Subject to Credit Risk." Journal of Finance, 50(1), 53–85.
https://doi.org/10.1111/j.1540-6261.1995.tb05167.x
(Reduced-form / intensity-based default framework.)

Duffie, D. & Singleton, K. J. (1999). "Modeling Term Structures of
Defaultable Bonds." Review of Financial Studies, 12(4), 687–720.
https://doi.org/10.1093/rfs/12.4.687
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Model parameters -------------------------------------------------------
mu = 0.05       # drift
sigma = 0.2     # volatility
lam = 0.5       # default intensity \lambda
S0 = 1.0        # initial asset value
T = 5.0         # final time
n = 1000        # number of steps

# --- Time grid and Brownian increments --------------------------------------
t = np.linspace(0.0, T, n + 1)
dt = t[1] - t[0]

rng = np.random.default_rng(seed=1)
dB = rng.normal(0.0, np.sqrt(dt), size=n)

# --- Sample default time ----------------------------------------------------
default_time = rng.exponential(1.0 / lam)
default_index = np.searchsorted(t, default_time) if default_time < T else None

# --- Initialize path --------------------------------------------------------
S = np.empty(n + 1)
S[0] = S0

# --- Euler–Maruyama before default -----------------------------------------
for i in range(n):
    if default_index is not None and i >= default_index:
        S[i + 1:] = 0.0
        break
    drift = mu * S[i] * dt
    diffusion = sigma * S[i] * dB[i]
    S[i + 1] = S[i] + drift + diffusion
else:
    # no default within the grid
    pass

# --- Plot ------------------------------------------------------------------
plt.plot(t, S)
if default_index is not None and default_index < len(t):
    plt.axvline(t[default_index], color="red", linestyle="--", label="default")
plt.xlabel("Time")
plt.ylabel("S(t)")
plt.title("Jump-to-default asset price sample path")
plt.legend()
plt.show()
