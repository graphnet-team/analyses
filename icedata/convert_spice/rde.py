# %%[markdown]
# # What does this script do?
#
# It takes the relative dom efficiency and build a look up table for doms.

# %% [markdown]
# # Imports
import importlib
import os
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

from utils import map_df_to_feature_map
import utils.geo
import utils.plot

# %% [markdown]
# # Load Data
# root of spice dir https://docs.icecube.aq/icetray/main/projects/ppc/index.html#configuration-files
root_source = Path('../ice-bfr-v2-x5-rde')
root_export = Path('../export/rde')

df_eff = pd.read_csv(root_source.joinpath('eff-f2k'), sep=' ',
                     names=['string_no', 'om_no', 'rde', 'type'])
utils.geo.attach_xyz_to_df(df_eff)

# pulsemap doms distinct
Z_SURFACE = 1948.07  # taken from https://wiki.icecube.wisc.edu/index.php/Coordinate_system

df_pulsemap_distinct = None
with sqlite3.connect('/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db') as con:
    sql = f'SELECT DISTINCT dom_x, dom_y, dom_z FROM SRTTWOfflinePulsesDC'
    df_pulsemap_distinct = pd.read_sql(sql, con)
assert df_pulsemap_distinct is not None  # for static typing
df_pulsemap_distinct['dom_z_global'] = df_pulsemap_distinct['dom_z'] - Z_SURFACE

# pulsemap doms
df_pulsemap = None
with sqlite3.connect('/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db') as con:
    sql = f'SELECT dom_x, dom_y, dom_z FROM SRTTWOfflinePulsesDC'
    df_pulsemap = pd.read_sql(sql, con)
assert df_pulsemap is not None  # for static typing
df_pulsemap['dom_z_global'] = df_pulsemap['dom_z'] - Z_SURFACE


# %% [markdown]
# # Build look up table
# all_dom_pos = list(df_pulsemap_distinct[['dom_x', 'dom_y', 'dom_z']].itertuples(index=False))
feature_map = {}
for x, y, z, z_global in df_pulsemap_distinct.itertuples(index=False):
    x_rounded, y_rounded, z_global_rounded = round(x, 2), round(y, 2), round(z_global, 2)

    row = df_eff[(df_eff['x'] == x_rounded) & (df_eff['y'] == y_rounded) & (df_eff['z'] == z_global_rounded)]

    feature_map[x, y, z] = row[['rde', 'type']].iloc[0].to_numpy()


# %% [markdown]
# # Make and fit Transformer
features = map_df_to_feature_map(df_pulsemap, feature_map)

transformer_robust = RobustScaler()
features_transformed_robust = transformer_robust.fit_transform(features)

transformer_quantile = QuantileTransformer(output_distribution='normal')
features_transformed_quantile = transformer_quantile.fit_transform(features)


# %% [markdown]
# # Make transformed look up table
feature_map_transformed_robust = {xyz: transformer_robust.transform(
    [features]).flatten() for xyz, features in feature_map.items()}
feature_map_transformed_quantile = {xyz: transformer_quantile.transform(
    [features]).flatten() for xyz, features in feature_map.items()}


# %% [markdown]
# # Write files/objects
root_export.mkdir(exist_ok=True)

with root_export.joinpath('feature_map.pkl').open('wb') as f:
    pickle.dump(feature_map, f)

with root_export.joinpath('feature_map_transformed_robust.pkl').open('wb') as f:
    pickle.dump(feature_map_transformed_robust, f)
with root_export.joinpath('transformer.pkl').open('wb') as f:
    pickle.dump(transformer_robust, f)

with root_export.joinpath('feature_map_transformed_quantile.pkl').open('wb') as f:
    pickle.dump(feature_map_transformed_quantile, f)
with root_export.joinpath('transformer_quantile.pkl').open('wb') as f:
    pickle.dump(transformer_quantile, f)

# %% [markdown]
# # Plot features
importlib.reload(utils.plot)

fig = utils.plot.figure_2d_doms_and_features(
    df_pulsemap_distinct,
    features_list=[
        map_df_to_feature_map(df_pulsemap_distinct, feature_map),
        map_df_to_feature_map(df_pulsemap_distinct, feature_map_transformed_robust),
        map_df_to_feature_map(df_pulsemap_distinct, feature_map_transformed_quantile),
    ],
    column_titles=['Raw', 'Transformed (RobustScaler)', 'Transformed (QuantileTransformer)'],
    row_titles=['rde', 'type'],
    title='Relative DOM Efficiency',
)
fig.show(renderer='jpg')

# %%
