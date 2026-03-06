"""Cox–Ingersoll–Ross (CIR) process simulation."""
import numpy as np
import plotly.graph_objects as go
from .common import make_layout, add_paths

METADATA = {
    "title": "Cox–Ingersoll–Ross (CIR)",
    "latex": r"dr_t = \kappa(\bar{r} - r_t)\,dt + \sigma\sqrt{r_t}\,dB_t",
    "description": (
        "The CIR model guarantees non-negative rates when the Feller condition "
        r"$2\kappa\bar{r} \geq \sigma^2$ holds. It is widely used for short-rate "
        "and stochastic-volatility modelling."
    ),
    "reference": (
        "- Cox, J. C., Ingersoll, J. E. & Ross, S. A. (1985). "
        "A Theory of the Term Structure of Interest Rates. "
        "*Econometrica*, 53(2), 385–407. https://doi.org/10.2307/1911242"
    ),
    "params": [
        {"key": "kappa", "label": "Mean-reversion speed κ", "min": 0.1,  "max": 10.0, "default": 2.0,  "step": 0.1},
        {"key": "r_bar", "label": "Long-run mean r̄",        "min": 0.001,"max": 0.3,  "default": 0.04, "step": 0.001, "format": "%.3f"},
        {"key": "sigma", "label": "Volatility σ",            "min": 0.001,"max": 0.5,  "default": 0.1,  "step": 0.005, "format": "%.3f"},
        {"key": "r0",    "label": "Initial rate r₀",         "min": 0.001,"max": 0.3,  "default": 0.02, "step": 0.001, "format": "%.3f"},
        {"key": "T",     "label": "Time horizon T",          "min": 0.25, "max": 20.0, "default": 5.0,  "step": 0.25},
    ],
    "max_paths": 20,
    "default_steps": 500,
    "max_steps": 2000,
}


def run(params: dict, n: int, n_paths: int, seed: int) -> go.Figure:
    kappa, r_bar, sigma, r0, T = (params["kappa"], params["r_bar"],
                                   params["sigma"], params["r0"], params["T"])
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n + 1)
    dt = t[1] - t[0]
    dB = rng.normal(0.0, np.sqrt(dt), (n_paths, n))

    paths = np.empty((n_paths, n + 1))
    paths[:, 0] = r0
    for i in range(n):
        r_pos = np.maximum(paths[:, i], 0.0)
        paths[:, i + 1] = (r_pos
                           + kappa * (r_bar - r_pos) * dt
                           + sigma * np.sqrt(r_pos) * dB[:, i])
        paths[:, i + 1] = np.maximum(paths[:, i + 1], 0.0)

    feller = "✅" if 2 * kappa * r_bar >= sigma ** 2 else "⚠️"
    title = f"CIR Model  {feller} Feller: 2κr̄={2*kappa*r_bar:.3f}, σ²={sigma**2:.3f}"

    fig = go.Figure()
    add_paths(fig, t, paths)
    fig.add_hline(y=r_bar, line_dash="dash", line_color="black",
                  annotation_text=f"r̄ = {r_bar:.3f}", annotation_position="right")
    fig.update_layout(**make_layout(title, "Time (years)", "r(t)"))
    return fig
