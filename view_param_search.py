# -*- coding: utf-8 -*-
"""Interative plotting of `param_search` results."""
import os
import sys
import argparse
import pandas as pd
import plotly.graph_objects as go

# Valiate command line args
parser = argparse.ArgumentParser(\
        description="Interactive plotting of parameter search")
parser.add_argument('label')
opts = parser.parse_args(sys.argv[1:])
filename = os.path.join("output","{}_param_search.csv".format(opts.label))

if not os.path.exists(filename):
    parser.error("Could not find parameter search summary data: %s" % filename)

df_summary = pd.read_csv(filename)

x_lower = round(0.9*min([df_summary['train'].min(), df_summary['val'].min()]),1)
x_upper = round(1.1*max([df_summary['train'].max(), df_summary['val'].max()]),1)


fig = go.Figure()
fig.add_trace(go.Scatter(
    x = df_summary['train'],
    y = df_summary['val'],
    text = df_summary['labels'],
    textposition = 'top center',
    mode='markers+text',
))
fig.update_xaxes(title="Training Error", range=[x_lower, x_upper])
fig.update_yaxes(title="Validation Error", range=[x_lower, x_upper])
fig.show()
