r"""Simulate the Heston stochastic volatility model.

The Heston model couples asset price \(S_t\) with a mean-reverting variance
process \(V_t\):

\[
    dS_t = \mu S_t\, dt + \sqrt{V_t}\, S_t\, dW_t^{(1)},
\]

\[
    dV_t = \kappa(\bar{v} - V_t)\, dt + \xi \sqrt{V_t}\, dW_t^{(2)},
\]

where \(W^{(1)}\) and \(W^{(2)}\) are correlated Brownian motions with

\[
    dW_t^{(1)}\, dW_t^{(2)} = \rho\, dt.
\]

Parameters:
- \(\mu\)      — drift of the asset,
- \(\kappa\)   — mean-reversion speed of variance,
- \(\bar{v}\)  — long-run variance,
- \(\xi\)      — volatility of variance (vol-of-vol),
- \(\rho\)     — correlation between asset and variance Brownian motions.

The Feller condition \(2\kappa\bar{v} \geq \xi^2\) ensures \(V_t > 0\) a.s.
This script uses the Euler–Maruyama scheme with full truncation for \(V_t\).

Seminal reference
-----------------
Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic
Volatility with Applications to Bond and Currency Options."
Review of Financial Studies, 6(2), 327–343.
https://doi.org/10.1093/rfs/6.2.327
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Model parameters -------------------------------------------------------
mu = 0.05       # asset drift
kappa = 2.0     # variance mean-reversion speed
v_bar = 0.04    # long-run variance (= 0.2^2)
xi = 0.3        # vol-of-vol
rho = -0.7      # correlation
V0 = 0.04       # initial variance
S0 = 100.0      # initial asset price
T = 1.0         # final time
n = 1000        # number of steps

# --- Feller condition check -------------------------------------------------
if 2 * kappa * v_bar < xi**2:
    print("Warning: Feller condition not satisfied; V_t may hit zero.")

# --- Time grid --------------------------------------------------------------
t = np.linspace(0.0, T, n + 1)
dt = t[1] - t[0]

# --- Correlated Brownian increments ----------------------------------------
rng = np.random.default_rng(seed=42)
Z1 = rng.standard_normal(n)
Z2 = rng.standard_normal(n)
dW1 = np.sqrt(dt) * Z1
dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

# --- Euler–Maruyama integration (full truncation for V) --------------------
S = np.empty(n + 1)
V = np.empty(n + 1)
S[0] = S0
V[0] = V0

for i in range(n):
    V_pos = max(V[i], 0.0)
    V[i + 1] = V[i] + kappa * (v_bar - V_pos) * dt + xi * np.sqrt(V_pos) * dW2[i]
    V[i + 1] = max(V[i + 1], 0.0)          # full truncation
    S[i + 1] = S[i] + mu * S[i] * dt + np.sqrt(V_pos) * S[i] * dW1[i]

# --- Plot -------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

ax1.plot(t, S, color="steelblue")
ax1.set_ylabel(r"$S_t$")
ax1.set_title("Heston stochastic volatility model")

ax2.plot(t, np.sqrt(V), color="darkorange", label=r"$\sqrt{V_t}$ (inst. vol)")
ax2.axhline(np.sqrt(v_bar), color="black", linestyle="--", linewidth=0.8,
            label=r"$\sqrt{\bar{v}}$ (long-run vol)")
ax2.set_xlabel("Time")
ax2.set_ylabel(r"$\sqrt{V_t}$")
ax2.legend()

plt.tight_layout()
plt.show()
