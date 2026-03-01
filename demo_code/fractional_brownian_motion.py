r"""Simulate fractional Brownian motion (fBm).

Fractional Brownian motion \(B^H_t\) is a zero-mean Gaussian process with
covariance

\[
    \mathbb{E}\!\left[B^H_t B^H_s\right]
    = \tfrac{1}{2}\!\left(t^{2H} + s^{2H} - |t - s|^{2H}\right),
\]

where \(H \in (0, 1)\) is the *Hurst exponent*:

- \(H = 0.5\)  — standard Brownian motion (independent increments).
- \(H > 0.5\)  — persistent (long-range dependent) increments.
- \(H < 0.5\)  — anti-persistent (negatively correlated) increments.

fBm is **not** a semi-martingale for \(H \neq 0.5\), so Itô calculus does
not apply directly.

**Simulation method (Cholesky decomposition):**
For a grid \(0 = t_0 < t_1 < \cdots < t_n = T\) the covariance matrix
\(\Sigma_{ij} = \tfrac{1}{2}(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})\)
is formed and factored via Cholesky; a sample path is then \(\Sigma^{1/2} Z\)
for \(Z \sim \mathcal{N}(0, I)\).  This is exact but \(O(n^2)\) in memory.

Seminal reference
-----------------
Mandelbrot, B. B. & Van Ness, J. W. (1968). "Fractional Brownian Motions,
Fractional Noises and Applications." SIAM Review, 10(4), 422–437.
https://doi.org/10.1137/1010093
"""

import numpy as np
import matplotlib.pyplot as plt


def fbm_cholesky(H: float, n: int, T: float = 1.0,
                 rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate a fractional Brownian motion path via Cholesky decomposition.

    Parameters
    ----------
    H : float
        Hurst exponent in (0, 1).
    n : int
        Number of time steps (path has n+1 points including t=0).
    T : float
        Terminal time.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    t : np.ndarray  shape (n+1,)
        Time grid including 0.
    B : np.ndarray  shape (n+1,)
        fBm sample path with B[0] = 0.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.linspace(0.0, T, n + 1)
    t_pos = t[1:]                             # exclude t=0 (B_0 = 0 a.s.)

    # Build covariance matrix on positive-time grid
    ti = t_pos[:, None]   # column
    tj = t_pos[None, :]   # row
    cov = 0.5 * (ti**(2 * H) + tj**(2 * H) - np.abs(ti - tj)**(2 * H))

    # Cholesky factor and sample
    L = np.linalg.cholesky(cov)
    Z = rng.standard_normal(n)
    B_pos = L @ Z

    B = np.concatenate([[0.0], B_pos])
    return t, B


# --- Simulation parameters --------------------------------------------------
T = 1.0
n = 500
rng = np.random.default_rng(seed=42)
hurst_values = [0.3, 0.5, 0.7]
colors = ["tab:blue", "tab:orange", "tab:green"]

# --- Plot -------------------------------------------------------------------
plt.figure(figsize=(9, 4))
for H, color in zip(hurst_values, colors):
    t, B = fbm_cholesky(H, n, T, rng=rng)
    plt.plot(t, B, color=color, alpha=0.85,
             label=rf"$H = {H}$")

plt.xlabel("Time")
plt.ylabel(r"$B^H_t$")
plt.title("Fractional Brownian motion — varying Hurst exponent")
plt.legend()
plt.tight_layout()
plt.show()
