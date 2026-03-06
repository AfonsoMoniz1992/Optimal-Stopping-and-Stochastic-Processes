"""Geometric Brownian Motion simulation."""
import numpy as np
import plotly.graph_objects as go
from .common import COLORS, make_layout, add_paths

METADATA = {
    "title": "Geometric Brownian Motion",
    "latex": r"dS_t = \mu S_t\,dt + \sigma S_t\,dB_t",
    "description": (
        "GBM is the standard continuous-time model for equity prices. "
        "Log-returns over any interval are normally distributed. "
        r"The exact solution is $S_t = S_0\exp\bigl((\mu-\tfrac{1}{2}\sigma^2)t + \sigma B_t\bigr)$."
    ),
    "reference": (
        "- Bachelier, L. (1900). *Théorie de la spéculation.* Annales Scientifiques de l'ENS, 17, 21–86.\n"
        "- Samuelson, P. A. (1965). Rational Theory of Warrant Pricing. *Industrial Management Review*, 6(2), 13–32.\n"
        "- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. "
        "*Journal of Political Economy*, 81(3), 637–654."
    ),
    "params": [
        {"key": "mu",    "label": "Drift μ",          "min": -0.5,  "max": 1.0,   "default": 0.10, "step": 0.01},
        {"key": "sigma", "label": "Volatility σ",     "min":  0.01, "max": 1.0,   "default": 0.20, "step": 0.01},
        {"key": "S0",    "label": "Initial price S₀", "min":  1.0,  "max": 500.0, "default": 100.0,"step": 5.0,  "format": "%.0f"},
        {"key": "T",     "label": "Time horizon T",   "min":  0.25, "max": 10.0,  "default": 1.0,  "step": 0.25},
    ],
    "max_paths": 20,
    "default_steps": 500,
    "max_steps": 2000,
}


def run(params: dict, n: int, n_paths: int, seed: int) -> go.Figure:
    mu, sigma, S0, T = params["mu"], params["sigma"], params["S0"], params["T"]
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n + 1)
    dt = t[1] - t[0]
    dB = rng.normal(0.0, np.sqrt(dt), (n_paths, n))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * dB
    log_paths = np.hstack([np.full((n_paths, 1), np.log(S0)),
                           np.cumsum(increments, axis=1)])
    paths = np.exp(log_paths)

    fig = go.Figure()
    add_paths(fig, t, paths)

    # Theoretical mean
    mean_line = S0 * np.exp(mu * t)
    fig.add_trace(go.Scatter(
        x=t, y=mean_line, mode="lines", name="E[S_t]",
        line=dict(color="black", width=2, dash="dash"), opacity=0.7,
    ))

    fig.update_layout(**make_layout("Geometric Brownian Motion", "Time (years)", "S(t)"))
    return fig
