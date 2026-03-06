"""Jump-to-Default (credit-risk) model simulation."""
import numpy as np
import plotly.graph_objects as go
from .common import COLORS, make_layout

METADATA = {
    "title": "Jump-to-Default Model",
    "latex": r"dS_t = (\mu + \lambda)\,S_t\,dt + \sigma S_t\,dB_t - S_{t^-}\,dN_t",
    "description": (
        "The firm's equity follows GBM between credit events. "
        "At a Poisson default time τ ~ Exp(λ) the share price jumps to zero. "
        "The drift is adjusted by λ so that the pre-default price is a local martingale "
        "under the risk-neutral measure."
    ),
    "reference": (
        "- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. "
        "*Journal of Financial Economics*, 3(1–2), 125–144. https://doi.org/10.1016/0304-405X(76)90022-2\n"
        "- Duffie, D. & Singleton, K. (1999). Modeling term structures of defaultable bonds. "
        "*Review of Financial Studies*, 12(4), 687–720."
    ),
    "params": [
        {"key": "mu",    "label": "Drift μ",             "min": -0.3, "max": 0.5,  "default": 0.05, "step": 0.01},
        {"key": "sigma", "label": "Volatility σ",        "min":  0.01,"max": 0.8,  "default": 0.2,  "step": 0.01},
        {"key": "lam",   "label": "Default intensity λ", "min":  0.01,"max": 3.0,  "default": 0.5,  "step": 0.01},
        {"key": "S0",    "label": "Initial price S₀",    "min":  1.0, "max": 500.0,"default": 100.0,"step": 5.0, "format": "%.0f"},
        {"key": "T",     "label": "Time horizon T",      "min":  0.25,"max": 10.0, "default": 3.0,  "step": 0.25},
    ],
    "max_paths": 20,
    "default_steps": 500,
    "max_steps": 2000,
}


def run(params: dict, n: int, n_paths: int, seed: int) -> go.Figure:
    mu, sigma, lam, S0, T = (params["mu"], params["sigma"],
                              params["lam"], params["S0"], params["T"])
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n + 1)
    dt = t[1] - t[0]

    # GBM increments (drift compensated for default)
    dB = rng.normal(0.0, np.sqrt(dt), (n_paths, n))
    # Default times: Exp(λ)
    default_times = rng.exponential(1.0 / lam, n_paths)

    paths = np.empty((n_paths, n + 1))
    paths[:, 0] = S0

    for p in range(n_paths):
        tau = default_times[p]
        for i in range(n):
            if t[i] >= tau:
                paths[p, i:] = 0.0
                break
            drift = (mu + lam) * paths[p, i] * dt
            diff = sigma * paths[p, i] * dB[p, i]
            paths[p, i + 1] = max(paths[p, i] + drift + diff, 0.0)
        else:
            # No default in [0, T]
            pass

    n_defaulted = int(np.sum(default_times <= T))
    title = f"Jump-to-Default  ({n_defaulted}/{n_paths} paths defaulted in [0, {T}])"

    fig = go.Figure()
    for i in range(n_paths):
        color = COLORS[i % len(COLORS)]
        defaulted = default_times[i] <= T
        fig.add_trace(go.Scatter(
            x=t, y=paths[i], mode="lines",
            name=f"Path {i + 1}" + (" ✗" if defaulted else ""),
            line=dict(color=color, width=1.5),
            opacity=0.8,
            showlegend=(n_paths <= 10),
        ))

    fig.update_layout(**make_layout(title, "Time (years)", "S(t)"))
    return fig
