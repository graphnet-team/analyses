# %% [markdown]
# # Imports
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import pickle

import sqlite3
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import QuantileTransformer, RobustScaler

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt


# %% [markdown]
# # Load data
df_truth = None
with sqlite3.connect('/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db') as con:
    query = f'SELECT * FROM truth WHERE abs(pid) != 13'
    df_truth = pd.read_sql(query, con)
assert df_truth is not None  # for static typing
df_truth['energy_log10'] = np.log10(df_truth['energy'])


# %% [markdown]
# # Plot original distribution
plt.figure(figsize=(8, 6))
ns, bins, patches = plt.hist(df_truth['energy_log10'], bins=100)
plt.show()


# %% [markdown]
# # Create weights
# Use original data distribution to flatten low
x_low = 1.5  # log10 GeV
alpha = 5


def calc_weights():
    weights = []
    for n, e in zip(ns, bins[:-1]):
        weights.append(
            1 / n * sum(ns[:16]) * ((x_low - e) * 0.15 + 1)
            if e < x_low else
            1 / (1 + alpha * (e - x_low))
        )
    return np.array(weights)


weights = calc_weights()


interp = interp1d(x=bins[:-1], y=weights, fill_value='extrapolate')
# normalize that avg weight is 1
avg_weight = np.average(interp(df_truth['energy_log10']))
print('Average weight of interp:', avg_weight)
interp = interp1d(x=bins[:-1], y=weights / avg_weight, fill_value='extrapolate')

# with open('loss_weight.pkl', 'wb') as f:
#     pickle.dump(interp, f)


# %% [markdown]
# # Plot adjusted distribution
plt.figure(figsize=(8, 6))
ns, bins, patches = plt.hist(df_truth['energy_log10'], bins=100, weights=interp(df_truth['energy_log10']))
plt.show()


# %% [markdown]
# # Plot weight
xs = np.arange(0, 4, 0.04)
ys = interp(xs)
plt.plot(xs, ys)
plt.show()


# %% [markdown]
# # Plot hist side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axes[0].hist(df_truth['energy_log10'], bins=100, density=True)
axes[1].hist(df_truth['energy_log10'], bins=100, density=True, weights=interp(df_truth['energy_log10']))
axes[0].set(xlabel='Energy [log10 GeV]', ylabel='Density')
axes[1].set(xlabel='Energy [log10 GeV]')
plt.show()

# %%
