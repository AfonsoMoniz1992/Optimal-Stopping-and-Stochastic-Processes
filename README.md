# Optimal Stopping and Stochastic Processes

This repository collects short Python scripts that demonstrate how to simulate and plot a few classical stochastic processes.  The code is intended for learners who want small, self‑contained examples of how to generate sample paths and visualize them.

## Repository Layout

- `demo_code/` – a set of independent demonstration scripts:
  - `euler_maruyama.py` – compares an exact geometric Brownian motion solution with the Euler–Maruyama approximation.
  - `euler_maruyama_v2.py` – simulates an Ornstein–Uhlenbeck process using Euler–Maruyama.
  - `geometric_brownian_motion_sullivan.py` – steps through Brownian increments, Brownian motion, and a geometric Brownian motion stock model.
  - `exemplo_plotly.py` – shows how to plot interactively with Plotly.
  - `allen-cahn_pde.py` – placeholder for future experiments with the Allen–Cahn partial differential equation.


## Common Stochastic Processes

### Brownian Motion
The basic building block for many stochastic models.  It has independent, normally distributed increments and continuous paths.  In Python you can simulate a path by drawing normal increments and cumulatively summing them to obtain the Brownian path.

### Geometric Brownian Motion
Used widely in mathematical finance to model stock prices.  It satisfies the stochastic differential equation

\[ dS_t = \mu S_t\,dt + \sigma S_t\,dB_t, \]

where \(\mu\) is the drift, \(\sigma\) the volatility, and \(B_t\) a standard Brownian motion.  The repository contains both a closed‑form simulation (`euler_maruyama.py`) and a more didactic walk‑through (`geometric_brownian_motion_sullivan.py`).

### Ornstein–Uhlenbeck Process
A mean‑reverting process that solves

\[ dX_t = \theta(\mu - X_t)dt + \sigma dB_t. \]

The script `euler_maruyama_v2.py` shows how to simulate it with the Euler–Maruyama scheme.


### Poisson Process
Counts the number of random arrivals over time.  It has stationary, independent
increments and is fully determined by its rate \(\lambda\).  The script
`poisson_process.py` draws exponential waiting times to build a sample path.

### Cox–Ingersoll–Ross Process
An interest rate model satisfying

\[ dR_t = \kappa(\theta - R_t)dt + \sigma \sqrt{R_t} dB_t. \]

The example `cox_ingersoll_ross.py` simulates this mean-reverting square-root
diffusion via Euler–Maruyama.


### Euler–Maruyama Method
A simple numerical integrator for SDEs.  Given a model

\[ dX_t = a(X_t,t)dt + b(X_t,t)dB_t, \]

the Euler–Maruyama update is

\[ X_{n+1} = X_n + a(X_n,t_n)\Delta t + b(X_n,t_n)\Delta B_n. \]

Both Euler–Maruyama examples in `demo_code/` use this formula.

## Running the Examples
Each script is self‑contained and can be executed directly, for example:

```bash
python demo_code/euler_maruyama.py
python demo_code/euler_maruyama_v2.py
python demo_code/geometric_brownian_motion_sullivan.py


The scripts produce plots using Matplotlib (and Plotly in `exemplo_plotly.py`).  In a headless environment you may need to set an appropriate Matplotlib backend, e.g. `MPLBACKEND=Agg`, to save plots instead of showing them interactively.

## Learn More
To deepen your understanding of these simulations, study the theory of stochastic calculus, including Ito calculus and numerical methods for SDEs.  Converting the scripts into reusable functions or Jupyter notebooks is a natural next step for exploration.

