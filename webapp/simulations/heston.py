"""Heston stochastic-volatility model simulation."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .common import COLORS, make_layout

METADATA = {
    "title": "Heston Stochastic Volatility",
    "latex": (
        r"dS_t = \mu S_t\,dt + \sqrt{V_t}\,S_t\,dW^S_t"
        r"\qquad"
        r"dV_t = \kappa(\bar{V} - V_t)\,dt + \xi\sqrt{V_t}\,dW^V_t"
    ),
    "description": (
        "The Heston model couples asset-price GBM with a mean-reverting CIR variance. "
        "The two Brownian motions are correlated: $dW^S_t\\,dW^V_t = \\rho\\,dt$. "
        "Negative ρ captures the *leverage effect* (prices fall → vol rises). "
        "The Feller condition $2\\kappa\\bar{V} \\geq \\xi^2$ keeps V > 0."
    ),
    "reference": (
        "- Heston, S. L. (1993). A Closed-Form Solution for Options with Stochastic Volatility "
        "with Applications to Bond and Currency Options. "
        "*Review of Financial Studies*, 6(2), 327–343. https://doi.org/10.1093/rfs/6.2.327"
    ),
    "params": [
        {"key": "mu",    "label": "Drift μ",                   "min": -0.2,  "max": 0.5,   "default":  0.05, "step": 0.01},
        {"key": "kappa", "label": "Vol mean-reversion κ",      "min":  0.1,  "max": 10.0,  "default":  2.0,  "step": 0.1},
        {"key": "v_bar", "label": "Long-run variance V̄",       "min":  0.01, "max": 0.25,  "default":  0.04, "step": 0.005, "format": "%.3f"},
        {"key": "xi",    "label": "Vol-of-vol ξ",              "min":  0.01, "max": 1.5,   "default":  0.3,  "step": 0.01},
        {"key": "rho",   "label": "Correlation ρ",             "min": -0.99, "max": 0.99,  "default": -0.7,  "step": 0.01},
        {"key": "V0",    "label": "Initial variance V₀",       "min":  0.001,"max": 0.25,  "default":  0.04, "step": 0.005, "format": "%.3f"},
        {"key": "S0",    "label": "Initial price S₀",          "min":  1.0,  "max": 500.0, "default": 100.0, "step": 5.0,   "format": "%.0f"},
        {"key": "T",     "label": "Time horizon T",            "min":  0.25, "max": 5.0,   "default":  1.0,  "step": 0.25},
    ],
    "max_paths": 15,
    "default_steps": 500,
    "max_steps": 2000,
}


def run(params: dict, n: int, n_paths: int, seed: int) -> go.Figure:
    mu, kappa, v_bar, xi, rho, V0, S0, T = (
        params["mu"], params["kappa"], params["v_bar"],
        params["xi"], params["rho"], params["V0"], params["S0"], params["T"],
    )
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n + 1)
    dt = t[1] - t[0]

    # Correlated Brownian increments
    Z1 = rng.normal(0.0, 1.0, (n_paths, n))
    Z2 = rng.normal(0.0, 1.0, (n_paths, n))
    dW_S = np.sqrt(dt) * Z1
    dW_V = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho ** 2) * Z2)

    S = np.empty((n_paths, n + 1))
    V = np.empty((n_paths, n + 1))
    S[:, 0] = S0
    V[:, 0] = V0

    for i in range(n):
        V_pos = np.maximum(V[:, i], 0.0)
        V[:, i + 1] = (V_pos
                       + kappa * (v_bar - V_pos) * dt
                       + xi * np.sqrt(V_pos) * dW_V[:, i])
        V[:, i + 1] = np.maximum(V[:, i + 1], 0.0)
        S[:, i + 1] = S[:, i] * (1.0 + mu * dt + np.sqrt(V_pos) * dW_S[:, i])
        S[:, i + 1] = np.maximum(S[:, i + 1], 0.0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Asset Price S(t)", "Variance V(t)"),
                        vertical_spacing=0.08, row_heights=[0.6, 0.4])

    for i in range(n_paths):
        color = COLORS[i % len(COLORS)]
        label = f"Path {i + 1}" if n_paths > 1 else "Path"
        show = n_paths <= 10
        fig.add_trace(go.Scatter(x=t, y=S[i], mode="lines", name=label,
                                  line=dict(color=color, width=1.5), opacity=0.8,
                                  showlegend=show, legendgroup=f"p{i}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=V[i], mode="lines", name=label,
                                  line=dict(color=color, width=1.2), opacity=0.7,
                                  showlegend=False, legendgroup=f"p{i}"), row=2, col=1)

    # Long-run variance line
    fig.add_hline(y=v_bar, line_dash="dash", line_color="black",
                  annotation_text=f"V̄={v_bar:.3f}", row=2, col=1)

    fig.update_layout(template="plotly_white", hovermode="x unified",
                      title=dict(text="Heston Stochastic Volatility", font=dict(size=18)),
                      margin=dict(l=60, r=20, t=60, b=60))
    fig.update_xaxes(title_text="Time (years)", row=2, col=1)
    fig.update_yaxes(title_text="S(t)", row=1, col=1)
    fig.update_yaxes(title_text="V(t)", row=2, col=1)
    return fig
