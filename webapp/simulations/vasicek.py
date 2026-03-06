"""Vasicek interest-rate model simulation."""
import numpy as np
import plotly.graph_objects as go
from .common import make_layout, add_paths

METADATA = {
    "title": "Vasicek Model",
    "latex": r"dr_t = \kappa(\theta - r_t)\,dt + \sigma\,dB_t",
    "description": (
        "The Vasicek model is the simplest Gaussian mean-reverting short-rate model. "
        "It admits closed-form bond prices but allows negative rates. "
        "The stationary distribution is $\\mathcal{N}(\\theta,\\,\\sigma^2/(2\\kappa))$."
    ),
    "reference": (
        "- Vasicek, O. (1977). An equilibrium characterisation of the term structure. "
        "*Journal of Financial Economics*, 5(2), 177–188. "
        "https://doi.org/10.1016/0304-405X(77)90016-2"
    ),
    "params": [
        {"key": "kappa", "label": "Mean-reversion speed κ", "min": 0.1,  "max": 10.0, "default": 1.5,  "step": 0.1},
        {"key": "theta", "label": "Long-run mean θ",         "min":-0.05, "max":  0.2, "default": 0.05, "step": 0.005, "format": "%.3f"},
        {"key": "sigma", "label": "Volatility σ",            "min": 0.001,"max":  0.1, "default": 0.02, "step": 0.001, "format": "%.3f"},
        {"key": "r0",    "label": "Initial rate r₀",         "min":-0.02, "max":  0.2, "default": 0.02, "step": 0.005, "format": "%.3f"},
        {"key": "T",     "label": "Time horizon T",          "min": 0.25, "max": 20.0, "default": 5.0,  "step": 0.25},
    ],
    "max_paths": 20,
    "default_steps": 500,
    "max_steps": 2000,
}


def run(params: dict, n: int, n_paths: int, seed: int) -> go.Figure:
    kappa, theta, sigma, r0, T = (params["kappa"], params["theta"],
                                   params["sigma"], params["r0"], params["T"])
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n + 1)
    dt = t[1] - t[0]
    dB = rng.normal(0.0, np.sqrt(dt), (n_paths, n))

    paths = np.empty((n_paths, n + 1))
    paths[:, 0] = r0
    for i in range(n):
        paths[:, i + 1] = (paths[:, i]
                           + kappa * (theta - paths[:, i]) * dt
                           + sigma * dB[:, i])

    fig = go.Figure()
    add_paths(fig, t, paths)
    fig.add_hline(y=theta, line_dash="dash", line_color="black",
                  annotation_text=f"θ = {theta:.3f}", annotation_position="right")
    fig.update_layout(**make_layout("Vasicek Model", "Time (years)", "r(t)"))
    return fig
