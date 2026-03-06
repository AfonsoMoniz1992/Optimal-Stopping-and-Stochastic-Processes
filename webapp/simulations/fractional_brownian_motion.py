"""Fractional Brownian Motion simulation via Cholesky decomposition."""
import numpy as np
import plotly.graph_objects as go
from .common import make_layout, add_paths

METADATA = {
    "title": "Fractional Brownian Motion",
    "latex": (
        r"\text{Cov}(B^H_s, B^H_t) = \tfrac{1}{2}"
        r"\bigl(|s|^{2H} + |t|^{2H} - |t-s|^{2H}\bigr)"
    ),
    "description": (
        "fBm generalises standard BM via the Hurst exponent **H ∈ (0, 1)**. "
        "H = 0.5 → standard BM. "
        "H > 0.5 → long-range dependence / persistence (trending). "
        "H < 0.5 → anti-persistence (mean-reverting increments). "
        "Simulated here via the exact Cholesky method on the covariance matrix."
    ),
    "reference": (
        "- Mandelbrot, B. B. & Van Ness, J. W. (1968). "
        "Fractional Brownian Motions, Fractional Noises and Applications. "
        "*SIAM Review*, 10(4), 422–437. https://doi.org/10.1137/1010093"
    ),
    "params": [
        {"key": "H", "label": "Hurst exponent H", "min": 0.05, "max": 0.95, "default": 0.7, "step": 0.05, "format": "%.2f"},
        {"key": "T", "label": "Time horizon T",   "min": 0.25, "max": 5.0,  "default": 1.0, "step": 0.25},
    ],
    "max_paths": 10,
    "default_steps": 200,
    "max_steps": 400,   # Cholesky is O(n^3); keep n small
}


def run(params: dict, n: int, n_paths: int, seed: int) -> go.Figure:
    H, T = params["H"], params["T"]
    rng = np.random.default_rng(seed)

    # Cholesky of the (n x n) covariance matrix
    t_pos = np.linspace(0.0, T, n + 1)[1:]      # shape (n,)
    ti = t_pos[:, None]
    tj = t_pos[None, :]
    cov = 0.5 * (np.abs(ti) ** (2 * H) + np.abs(tj) ** (2 * H) - np.abs(ti - tj) ** (2 * H))
    cov += 1e-10 * np.eye(n)                      # numerical regularisation
    L = np.linalg.cholesky(cov)

    # Simulate n_paths at once: L @ Z, Z ~ N(0, I_{n x n_paths})
    Z = rng.standard_normal((n, n_paths))
    B_pos = (L @ Z).T                              # (n_paths, n)
    paths = np.hstack([np.zeros((n_paths, 1)), B_pos])

    t = np.linspace(0.0, T, n + 1)
    fig = go.Figure()
    add_paths(fig, t, paths)
    fig.add_hline(y=0.0, line_dash="dot", line_color="black", line_width=1)
    fig.update_layout(**make_layout(
        f"Fractional Brownian Motion (H = {H:.2f})",
        "Time", "B^H(t)",
    ))
    return fig
