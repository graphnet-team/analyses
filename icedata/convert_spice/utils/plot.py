from typing import Iterable, List
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

SW_225 = np.array([-1, -1]) / np.sqrt(2)  # normalized direction vector


def sw_225(xy, *, origin=np.array([0, 0])):
    '''Returns length of `xy` which is parallel to SouthWest-225deg.'''
    return np.dot(xy - origin, SW_225)


def scatter_2d_doms(
    df: pd.DataFrame,
    *,
    values: Iterable,
    x_col='dom_x',
    y_col='dom_y',
    z_col='dom_z_global',
):
    sws = sw_225(df[[x_col, y_col]])
    zs = df[z_col]

    return go.Scatter(
        x=sws, y=zs,
        mode='markers',
        marker=dict(color=values, size=4) if values is not None else dict(size=4),
        text=values,
    )


def figure_2d_doms_and_features(
    doms,
    features_list: List,
    *,
    column_titles: List[str],
    row_titles: List[str],
    title='',
):
    rows = features_list[0].shape[1]
    cols = len(features_list)
    fig = make_subplots(
        rows=rows, cols=cols,
        column_titles=column_titles,
        row_titles=row_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )
    fig.update_layout(
        title=title,
        width=500 * cols,
        height=400 * rows,
        showlegend=False,
    )
    for x, features in enumerate(features_list):
        for y in range(features.shape[1]):
            feature = features[:, y]
            fig.add_trace(
                scatter_2d_doms(doms, values=feature),
                col=x+1,
                row=y+1,
            )
            fig.update_xaxes(showgrid=False, zeroline=False)
            fig.update_yaxes(showgrid=False, zeroline=False)
            if x == 0:
                fig.update_yaxes(title_text="z_global", row=y+1, col=x+1)
            if y == features.shape[1]-1:
                fig.update_xaxes(title_text="Direction SW-225deg", row=y+1, col=x+1)

    return fig
