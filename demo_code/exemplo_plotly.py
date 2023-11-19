"""
Exemplo Plotly Offline

"""

import numpy as np
import plotly as py
import plotly.offline as pyo
import plotly.graph_objs as go
import ipywidgets as widgets

from scipy import special

pyo.init_notebook_mode()


x=np.linspace(0,np.pi,1000)
layout=go.Layout(title='Simple Example',yaxis=dict(title='volts'),xaxis=dict(title='nanoseconds'))
trace1=go.Scatter(x=x,y=np.sin(x),mode='lines',name='sin(x)',line=dict(shape='spline')) 
fig=go.Figure(data=[trace1], layout=layout)
pyo.iplot(fig)