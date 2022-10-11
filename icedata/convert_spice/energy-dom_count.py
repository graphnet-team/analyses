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

df_pulsemap = None
with sqlite3.connect('/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db') as con:
    query = f'SELECT * FROM SRTTWOfflinePulsesDC'
    df_pulsemap = pd.read_sql(query, con)
assert df_pulsemap is not None  # for static typing


df_truth = None
with sqlite3.connect('/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db') as con:
    query = f'SELECT * FROM truth WHERE abs(pid) != 13'
    df_truth = pd.read_sql(query, con)
assert df_truth is not None  # for static typing


# %% [markdown]
# # Count pulses per event
pulses_of_events_series = df_pulsemap['event_no'].value_counts()
pulses_of_events = pd.DataFrame({'event_no': pulses_of_events_series.index, 'count': pulses_of_events_series.values})


# %% [markdown]
# # Join count table with truth energy
df = pulses_of_events.set_index('event_no').join(df_truth[['event_no', 'energy']].set_index('event_no'), how='inner')
df['energy_log10'] = np.log10(df['energy'])
# df has weird format.... idk.. index name is below cols... ?

# %% [markdown]
# # Plot heatmap
# fig = px.density_heatmap(df.iloc[:13000], x="count", y="energy", log_y=True)

# fig.show(renderer='jpg')
# fig.write_image('energy-dom_count.jpg')
plt.hist2d(df['count'], df['energy_log10'], bins=60, range=[[1, 60], [0, 2.5]])
plt.xlabel('Pulses per event')
plt.ylabel('Energy log10 [GeV]')
plt.savefig('energy-dom_count.jpg', dpi=600)
plt.show()

# %%
