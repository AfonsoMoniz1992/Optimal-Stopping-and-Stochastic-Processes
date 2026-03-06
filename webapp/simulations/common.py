"""Shared utilities for simulation modules."""
import plotly.graph_objects as go
from plotly.colors import qualitative

COLORS = qualitative.Plotly  # 10 distinct colors


def make_layout(title: str, xaxis_title: str, yaxis_title: str, **kwargs) -> dict:
    layout_kwargs = dict(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    layout_kwargs.update(kwargs)
    return layout_kwargs


def add_paths(fig: go.Figure, t, paths, name_prefix: str = "Path",
              colors=None, opacity: float = 0.8, row=None, col=None):
    """Add multiple simulation paths to a figure."""
    if colors is None:
        colors = COLORS
    n_paths = paths.shape[0]
    kw = {}
    if row is not None:
        kw["row"] = row
        kw["col"] = col
    for i in range(n_paths):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=t,
            y=paths[i],
            mode="lines",
            name=f"{name_prefix} {i + 1}" if n_paths > 1 else name_prefix,
            line=dict(color=color, width=1.5),
            opacity=opacity,
            showlegend=(n_paths <= 10),
        ), **kw)
    return fig
