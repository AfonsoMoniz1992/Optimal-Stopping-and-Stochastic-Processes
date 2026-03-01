r"""Simulate the Merton jump-diffusion model for asset prices.

Merton's model extends geometric Brownian motion by adding a compound Poisson
jump component:

\[
    \frac{dS_t}{S_{t-}} = (\mu - \lambda \bar{k})\, dt
        + \sigma\, dB_t + (J_t - 1)\, dN_t,
\]

where
- \(\mu\)          — drift,
- \(\sigma\)        — diffusion volatility,
- \(B_t\)           — standard Brownian motion,
- \(N_t\)           — Poisson process with intensity \(\lambda\),
- \(J_t - 1\)       — random jump size; \(\ln J_t \sim \mathcal{N}(\mu_J, \sigma_J^2)\),
- \(\bar{k} = e^{\mu_J + \sigma_J^2/2} - 1\) — mean jump size (drift correction).

The exact solution at each step between jumps is a GBM; jumps are applied
multiplicatively at random arrival times.

Seminal reference
-----------------
Merton, R. C. (1976). "Option pricing when underlying stock returns are
discontinuous." Journal of Financial Economics, 3(1–2), 125–144.
https://doi.org/10.1016/0304-405X(76)90022-2
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Model parameters -------------------------------------------------------
mu = 0.05       # drift (before jump correction)
sigma = 0.2     # diffusion volatility
lam = 1.0       # Poisson jump intensity (expected jumps per unit time)
mu_J = -0.1     # mean of log-jump size
sigma_J = 0.15  # std dev of log-jump size
S0 = 100.0      # initial asset price
T = 1.0         # final time
n = 1000        # number of time steps

# --- Drift correction -------------------------------------------------------
k_bar = np.exp(mu_J + 0.5 * sigma_J**2) - 1.0  # mean of (J - 1)

# --- Time grid --------------------------------------------------------------
t = np.linspace(0.0, T, n + 1)
dt = t[1] - t[0]

# --- Brownian increments and Poisson arrivals -------------------------------
rng = np.random.default_rng(seed=42)
dB = rng.normal(0.0, np.sqrt(dt), size=n)
poisson_counts = rng.poisson(lam * dt, size=n)   # N_{t+dt} - N_t per step

# --- Euler–Maruyama with exact Poisson jump application --------------------
S = np.empty(n + 1)
S[0] = S0

for i in range(n):
    # Brownian (GBM) increment
    gbm = (mu - lam * k_bar - 0.5 * sigma**2) * dt + sigma * dB[i]

    # Compound jump: product of log-normal jumps that arrive in this step
    m = poisson_counts[i]
    if m > 0:
        log_jumps = rng.normal(mu_J, sigma_J, size=m)
        jump_factor = np.exp(np.sum(log_jumps))
    else:
        jump_factor = 1.0

    S[i + 1] = S[i] * np.exp(gbm) * jump_factor

# --- Plot -------------------------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(t, S, label="Merton jump-diffusion")
plt.xlabel("Time")
plt.ylabel(r"$S_t$")
plt.title("Merton jump-diffusion model")
plt.legend()
plt.tight_layout()
plt.show()
