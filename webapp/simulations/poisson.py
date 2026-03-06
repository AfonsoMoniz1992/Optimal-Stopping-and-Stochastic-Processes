"""Poisson process simulation."""
import numpy as np
import plotly.graph_objects as go
from .common import COLORS, make_layout

METADATA = {
    "title": "Poisson Process",
    "latex": r"N(t) \sim \text{Poisson}(\lambda t),\quad \lambda > 0",
    "description": (
        "The Poisson process counts random events arriving at a constant rate λ. "
        "Inter-arrival times are i.i.d. Exp(λ). It is the foundation for "
        "compound Poisson models, jump-diffusions, and credit-default intensity models."
    ),
    "reference": (
        "- Poisson, S. D. (1837). *Recherches sur la probabilité des jugements.* Paris: Bachelier.\n"
        "- Ross, S. M. (1996). *Stochastic Processes* (2nd ed.). Wiley."
    ),
    "params": [
        {"key": "lam", "label": "Intensity λ (events/unit time)", "min": 0.1, "max": 20.0, "default": 3.0, "step": 0.1},
        {"key": "T",   "label": "Time horizon T",                 "min": 0.5, "max": 20.0, "default": 5.0, "step": 0.5},
    ],
    "max_paths": 15,
    "default_steps": 1,   # not used; internal grid is fixed
    "max_steps": 1,
}


def run(params: dict, n: int, n_paths: int, seed: int) -> go.Figure:
    lam, T = params["lam"], params["T"]
    rng = np.random.default_rng(seed)

    t_grid = np.linspace(0.0, T, 2000)
    fig = go.Figure()

    for i in range(n_paths):
        color = COLORS[i % len(COLORS)]
        # Sample jump times via exponential inter-arrivals
        arrivals = []
        s = 0.0
        while True:
            s += rng.exponential(1.0 / lam)
            if s > T:
                break
            arrivals.append(s)

        # Build step-function path
        jump_times = np.array([0.0] + arrivals + [T])
        counts = np.arange(len(jump_times))

        # Step function: for each segment hold count constant
        xs, ys = [0.0], [0]
        for j in range(1, len(jump_times)):
            xs += [jump_times[j], jump_times[j]]
            ys += [counts[j - 1], counts[j]]
        xs.append(T)
        ys.append(counts[-1])

        label = f"Path {i + 1}" if n_paths > 1 else "N(t)"
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=label,
            line=dict(color=color, width=1.5, shape="hv"),
            opacity=0.8,
            showlegend=(n_paths <= 10),
        ))

    # Theoretical mean E[N(t)] = λt
    fig.add_trace(go.Scatter(
        x=[0.0, T], y=[0.0, lam * T], mode="lines",
        name=f"E[N(t)] = λt",
        line=dict(color="black", width=2, dash="dash"),
    ))

    fig.update_layout(**make_layout("Poisson Process", "Time", "N(t)  (count)"))
    fig.update_yaxes(rangemode="tozero")
    return fig
