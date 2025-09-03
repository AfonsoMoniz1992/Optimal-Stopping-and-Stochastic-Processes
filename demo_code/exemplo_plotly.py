r"""Offline Plotly example plotting the function :math:`\sin(x)` on
\([0, \pi]\).

Demonstrates how to build an interactive figure with Plotly's offline
mode.
"""

import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

# Activate offline mode (useful in Jupyter notebooks)
pyo.init_notebook_mode()

# Sample points and compute \sin(x)
x = np.linspace(0, np.pi, 1000)
y = np.sin(x)

# Build the figure
title = "Simple Example"
layout = go.Layout(
    title=title, yaxis=dict(title="volts"), xaxis=dict(title="nanoseconds")
)
trace = go.Scatter(x=x, y=y, mode="lines", name="sin(x)", line=dict(shape="spline"))
fig = go.Figure(data=[trace], layout=layout)

# Display the plot
pyo.iplot(fig)
