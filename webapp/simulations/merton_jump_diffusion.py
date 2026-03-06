"""Merton Jump-Diffusion model simulation."""
import numpy as np
import plotly.graph_objects as go
from .common import make_layout, add_paths

METADATA = {
    "title": "Merton Jump-Diffusion",
    "latex": (
        r"dS_t = (\mu - \lambda\bar{k})\,S_t\,dt + \sigma S_t\,dB_t "
        r"+ S_{t^-}(e^{J_i}-1)\,dN_t"
    ),
    "description": (
        "Merton extended GBM with a compound Poisson jump term. "
        "Jump sizes are log-normally distributed: $\\ln(J_i)\\sim\\mathcal{N}(\\mu_J,\\sigma_J^2)$, "
        "so $\\bar{k}=e^{\\mu_J+\\sigma_J^2/2}-1$. "
        "The drift is compensated to keep the process a martingale under Q."
    ),
    "reference": (
        "- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. "
        "*Journal of Financial Economics*, 3(1–2), 125–144. "
        "https://doi.org/10.1016/0304-405X(76)90022-2"
    ),
    "params": [
        {"key": "mu",      "label": "Drift μ",             "min": -0.3, "max": 0.5,  "default":  0.05, "step": 0.01},
        {"key": "sigma",   "label": "Diffusion σ",         "min":  0.01,"max": 0.8,  "default":  0.2,  "step": 0.01},
        {"key": "lam",     "label": "Jump intensity λ",    "min":  0.0, "max": 10.0, "default":  1.0,  "step": 0.1},
        {"key": "mu_J",    "label": "Mean log-jump μ_J",   "min": -1.0, "max": 1.0,  "default": -0.1,  "step": 0.01},
        {"key": "sigma_J", "label": "Log-jump vol σ_J",    "min":  0.01,"max": 0.8,  "default":  0.15, "step": 0.01},
        {"key": "S0",      "label": "Initial price S₀",    "min":  1.0, "max": 500.0,"default": 100.0, "step": 5.0, "format": "%.0f"},
        {"key": "T",       "label": "Time horizon T",      "min":  0.25,"max": 5.0,  "default":  1.0,  "step": 0.25},
    ],
    "max_paths": 20,
    "default_steps": 500,
    "max_steps": 2000,
}


def run(params: dict, n: int, n_paths: int, seed: int) -> go.Figure:
    mu, sigma, lam, mu_J, sigma_J, S0, T = (
        params["mu"], params["sigma"], params["lam"],
        params["mu_J"], params["sigma_J"], params["S0"], params["T"],
    )
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, T, n + 1)
    dt = t[1] - t[0]

    k_bar = np.exp(mu_J + 0.5 * sigma_J ** 2) - 1.0   # E[e^J - 1]
    adj_mu = mu - lam * k_bar

    # Diffusion increments
    dB = rng.normal(0.0, np.sqrt(dt), (n_paths, n))

    # Poisson jump counts per step
    N_jumps = rng.poisson(lam * dt, (n_paths, n))

    paths = np.empty((n_paths, n + 1))
    paths[:, 0] = S0
    for i in range(n):
        # Log-normal jump aggregate per step
        total_log_jump = np.where(
            N_jumps[:, i] > 0,
            mu_J * N_jumps[:, i] + sigma_J * np.sqrt(N_jumps[:, i]) * rng.standard_normal(n_paths),
            0.0,
        )
        log_inc = (adj_mu - 0.5 * sigma ** 2) * dt + sigma * dB[:, i] + total_log_jump
        paths[:, i + 1] = paths[:, i] * np.exp(log_inc)

    fig = go.Figure()
    add_paths(fig, t, paths)
    # Theoretical mean under original measure
    mean_line = S0 * np.exp(mu * t)
    fig.add_trace(go.Scatter(
        x=t, y=mean_line, mode="lines", name="E[S_t]",
        line=dict(color="black", width=2, dash="dash"), opacity=0.7,
    ))
    fig.update_layout(**make_layout("Merton Jump-Diffusion", "Time (years)", "S(t)"))
    return fig
