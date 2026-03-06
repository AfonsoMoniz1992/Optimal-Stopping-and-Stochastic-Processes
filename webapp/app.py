"""Stochastic Process Interactive Simulator — Streamlit web app."""
import importlib
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stochastic Process Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Process registry ──────────────────────────────────────────────────────────
PROCESS_REGISTRY = {
    "Geometric Brownian Motion":        "gbm",
    "Ornstein–Uhlenbeck":               "ornstein_uhlenbeck",
    "Cox–Ingersoll–Ross (CIR)":         "cir",
    "Vasicek":                          "vasicek",
    "Poisson Process":                  "poisson",
    "Jump-to-Default":                  "jump_to_default",
    "Heston Stochastic Volatility":     "heston",
    "Merton Jump-Diffusion":            "merton_jump_diffusion",
    "Fractional Brownian Motion":       "fractional_brownian_motion",
}


@st.cache_resource(show_spinner=False)
def load_module(module_key: str):
    return importlib.import_module(f"simulations.{module_key}")


@st.cache_data(max_entries=256, show_spinner="Simulating…")
def cached_run(module_key: str, params_tuple: tuple, n_steps: int, n_paths: int, seed: int):
    mod = load_module(module_key)
    return mod.run(dict(params_tuple), n_steps, n_paths, seed)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    process_name = st.selectbox("**Process**", list(PROCESS_REGISTRY.keys()), index=0)
    module_key = PROCESS_REGISTRY[process_name]
    mod = load_module(module_key)
    meta = mod.METADATA

    st.markdown("---")
    st.markdown("**Model parameters**")

    params: dict = {}
    for p in meta["params"]:
        is_int = p.get("type") == "int"
        cast = int if is_int else float
        kwargs = dict(
            label=p["label"],
            min_value=cast(p["min"]),
            max_value=cast(p["max"]),
            value=cast(p["default"]),
            step=cast(p.get("step", 1 if is_int else 0.01)),
        )
        if not is_int:
            kwargs["format"] = p.get("format", "%.2f")
        params[p["key"]] = st.slider(**kwargs)

    st.markdown("---")
    st.markdown("**Simulation settings**")

    max_paths = meta.get("max_paths", 20)
    n_paths = st.slider("Number of paths", 1, max_paths, min(5, max_paths))

    default_steps = meta.get("default_steps", 500)
    max_steps = meta.get("max_steps", 2000)

    # Hide the n_steps slider for Poisson (uses its own internal grid)
    if module_key != "poisson":
        n_steps = st.slider("Time steps", 100, max_steps, default_steps, step=100)
    else:
        n_steps = 1  # unused for Poisson

    seed = int(st.number_input("Random seed", min_value=0, max_value=99_999,
                               value=42, step=1))

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown(f"# {meta['title']}")
st.latex(meta["latex"])

# Run simulation (result is cached based on all inputs)
params_tuple = tuple(sorted(params.items()))
fig = cached_run(module_key, params_tuple, n_steps, n_paths, seed)
st.plotly_chart(fig, use_container_width=True)

# Info panels
col_desc, col_ref = st.columns(2)
with col_desc:
    with st.expander("📐 Model description", expanded=True):
        st.markdown(meta["description"])

with col_ref:
    with st.expander("📚 Seminal references", expanded=True):
        st.markdown(meta["reference"])
