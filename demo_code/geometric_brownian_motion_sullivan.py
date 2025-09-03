r"""
Geometric Brownian motion example based on John Sullivan's tutorial.

We generate Brownian motion increments and simulate the exact solution to the
stochastic differential equation

\[
    dS_t = \mu S_t\, dt + \sigma S_t\, dB_t,
\]

whose closed form is

\[
    S_t = S_0 \exp\left((\mu - \tfrac{1}{2}\sigma^2)t + \sigma B_t\right).
\]
"""

import numpy as np
import matplotlib.pyplot as plt

# Ensure deterministic output and clear previous figures
plt.close("all")
seed = 1
N = 1000  # number of increments

def brownian(seed: int, N: int):
    """Generate Brownian increments and path.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    N : int
        Number of increments.

    Returns
    -------
    W : np.ndarray
        Brownian path.
    b : np.ndarray
        Increments \(\Delta B\).
    dt : float
        Time step size.
    """
    np.random.seed(seed)
    dt = 1.0 / N
    b = np.random.normal(0.0, 1.0, N) * np.sqrt(dt)
    W = np.cumsum(b)
    return W, b, dt

# Brownian motion and increments
W, b, dt = brownian(seed, N)
W = np.insert(W, 0, 0.0)  # start at 0

# Plot Brownian increments
plt.rcParams["figure.figsize"] = (10, 8)
xb = np.linspace(0, len(b), len(b))
plt.plot(xb, b)
plt.title("Brownian increments \u0394B")
plt.show()

# Plot Brownian path
xw = np.linspace(0, len(W), len(W))
plt.plot(xw, W)
plt.title("Standard Brownian motion B_t")
plt.show()

def GBM(S0, mu, sigma, W, T, N):
    """Exact solution for geometric Brownian motion."""
    t = np.linspace(0.0, 1.0, N + 1)
    S = [S0]
    for i in range(1, N + 1):
        drift = (mu - 0.5 * sigma**2) * t[i]
        diffusion = sigma * W[i - 1]
        S.append(S0 * np.exp(drift + diffusion))
    return np.array(S), t

S0 = 55.25  # initial stock price
mu = 0.15   # drift coefficient
sigma = 0.4 # volatility
solution, t = GBM(S0, mu, sigma, W, N + 1, N)

plt.plot(t, solution)
plt.ylabel("Stock Price ($)")
plt.title("Geometric Brownian Motion")
plt.show()
