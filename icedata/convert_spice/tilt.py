# %% [markdown]
# # What does this script do?
#
# It takes the icedata from spice and builds 4 lookup tables for doms.
#
# 1. icedata tilted layers
# 2. icedata tilted layers + transformer
# 3. icedata parallel layers
# 4. icedata parallel layers + transformer
#
# The lookup tables are pickled dict files which map a tuple (x,y,z) to a list of features.


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


# %% [markdown]
# # Load Data
# root of spice dir https://docs.icecube.aq/icetray/main/projects/ppc/index.html#configuration-files
root = Path('../ice-bfr-v2-x5-rde')
EXPORT_DIR = '../export'

# tilt
df_tilt_dat = pd.read_csv(root.joinpath('tilt.dat'), sep=' ',
                          names=['z', 'delta_z_0', 'delta_z_1', 'delta_z_2', 'delta_z_3', 'delta_z_4', 'delta_z_5'])
df_tilt_par = pd.read_csv(root.joinpath('tilt.par'), sep=' ',
                          names=['string_no', 'dist_sw_225'])

# geo/doms
df_geo = pd.read_csv(root.joinpath('geo-f2k'), sep='\t',
                     names=['dom_id', 'mainboard_id', 'x', 'y', 'z', 'string_no', 'om_no'])
df_geo_filtered = df_geo[df_geo.z < -5]  # filter out surface

# icemodel
columns_icemodel = ['be(400)', 'adust(400)', 'k1', 'k2', 'BFRA', 'BFRB']
COLUMNS_ICEMODEL_INTERESTING = [0, 1, 2, 5]
df_icemodel = pd.read_csv(root.joinpath('icemodel.dat'), sep=' ',
                          names=['z'] + columns_icemodel)
df_icemodel = df_icemodel.iloc[:-1, :]  # skip last row, B is many times higher than others (1036.2006 instead of 0.0X)

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

# strings
XY_STRING = {}
for row in df_geo_filtered.itertuples(index=False):
    if row.string_no not in XY_STRING:
        XY_STRING[row.string_no] = np.array([row.x, row.y])

# stings which are on the outside/border of icecube
# OUTSIDE_STRING_NOS = [1, 2, 3, 4, 5, 6, 13, 21, 30, 40, 50, 59, 67, 74, 78, 77, 76, 75, 68, 60, 51, 41, 31, 22, 14, 7]


# %% [markdown]
# # Define geometric
SW_225 = np.array([-1, -1]) / np.sqrt(2)  # normalized direction vector
ORIGIN = XY_STRING[86]  # string 86 is zero of `tilt.dat`


def sw_225(xy):
    '''Returns length of `xy` which is parallel to SouthWest-225deg.'''
    return np.dot(xy - ORIGIN, SW_225)


# %% [markdown]
# # Make DOM icedata lookup table
# Get icedata from `icemodel.dat` for each layer
# Interpolate the icedata based off layer_zs and dom_z_global
interpolation_dzs = interp1d(
    df_tilt_par.dist_sw_225.to_numpy(),  # locations of recorded dzs (6,)
    df_tilt_dat.iloc[:, 1:].to_numpy(),  # values of recorded dzs (125, 6)
    kind='quadratic',
    axis=1,
    bounds_error=False,
    fill_value='extrapolate',  # fill_value=(df_tilt_dat.iloc[:, 1], df_tilt_dat.iloc[:, -1]),  # cap values, skip z
)
def interpolation_zs(sw): return -(df_tilt_dat.z - interpolation_dzs(sw))
def interpolation_zs_0(sw): return -df_tilt_dat.z


def compute_feature_map(interpolation_zs):
    features_map = dict()
    for dom_x, dom_y, dom_z, dom_z_global in df_pulsemap_distinct.itertuples(index=False):
        dom_xy = np.array([dom_x, dom_y])  # (2,)
        dom_sw = sw_225(dom_xy)  # float

        # interpolate dz at xy/sw for all layers
        layer_zs_global = interpolation_zs(dom_sw)  # (125,)

        # calc features
        layer_features = df_icemodel[df_icemodel['z'].isin(df_tilt_dat['z'])][columns_icemodel].to_numpy()
        interpolation_features = interp1d(
            layer_zs_global,  # (125,)
            layer_features,  # (125, 6)
            kind='quadratic',
            axis=0,
            bounds_error=False,
            fill_value='extrapolate',  # (layer_features[-1], layer_features[0])
        )

        # assert that interpolation is correct for at all values
        # for zs, features in zip(layer_zs_global, layer_features):
        #     assert np.allclose(interpolation_features(zs), features)

        # get interpolated features for this node
        features_map[dom_x, dom_y, dom_z] = interpolation_features(dom_z_global)  # (6,)
    return features_map


feature_map = compute_feature_map(interpolation_zs)
feature_map_flat = compute_feature_map(interpolation_zs_0)


# %% [markdown]
# # Make 3D and 2D plots of doms and layers
def plot_3d_layers(interpolation_zs):
    dom_xys = np.array(list(XY_STRING.values()))  # (87, 2)
    dom_sws = sw_225(dom_xys)  # (87,)
    layer_zs = np.array([interpolation_zs(dom_sw) for dom_sw in dom_sws])  # (87, 125)

    data = []
    for zs in layer_zs.T:
        data.append(go.Mesh3d(
            x=dom_xys[:, 0], y=dom_xys[:, 1], z=zs,
            intensity=zs,
            hovertext=f'Layer center @ {zs}',
        ))

    # scatter plot doms from `geo-f2k`
    data.append(go.Scatter3d(
        x=df_geo_filtered.x, y=df_geo_filtered.y, z=df_geo_filtered.z,
        text=f'string {df_geo_filtered.string_no}',
        mode='markers',
        marker=dict(color='black', size=2),
        textposition='top center',
    ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title="Layer Overview (vertices evaluated at each dom xy positions)",
        width=1500,
        height=1500,
    )
    return fig


def plot_2d_layers(interpolation_zs):
    sws = np.arange(-600, 600)
    # sws = [-600, *df_tilt_par.dist_sw_225, 600]  # show extrapolation/caping
    layer_zs = np.array([interpolation_zs(sw) for sw in sws])  # (87, 125)

    data = []
    for zs in layer_zs.T:
        data.append(go.Scatter(
            x=sws, y=zs,
            hovertext=f'Layer center @ {zs}',
            # mode='lines',
            # line=dict(color='lightblue'),
        ))

    # scatter plot doms from `geo-f2k`
    data.append(go.Scatter(
        x=sw_225(np.array([df_geo_filtered.x, df_geo_filtered.y]).T), y=df_geo_filtered.z,
        text=f'string {df_geo_filtered.string_no}',
        mode='markers',
        marker=dict(color='black', size=2),
        textposition='top center',
    ))

    fig = go.Figure(data=data)
    fig.update_layout(
        # title="Layer Overview",
        xaxis_title="Distance in SW_225 direction (0 at string 86)",
        yaxis_title="Z",
        width=700,
        height=700,
        showlegend=False,
    )
    return fig


def plot_2d_and_3d_layers(interpolation_zs):
    fig_3d = plot_3d_layers(interpolation_zs)
    fig_3d.show()
    fig_3d.write_html(f'{EXPORT_DIR}/layers_3d.html')

    fig_2d = plot_2d_layers(interpolation_zs)
    fig_2d.show()
    fig_2d.write_html(f'{EXPORT_DIR}/layers_2d.html')
    fig_2d.write_image(f'{EXPORT_DIR}/layers_2d.jpg')


plot_2d_and_3d_layers(interpolation_zs)
# plot_2d_and_3d_layers(interpolation_zs_0)

# %% [markdown]
# # Assertions
# Assert that every dom in the pulsemap has a corresponding dom on `geo-f2k`.
all_dom_pos = list(df_geo_filtered[['x', 'y', 'z']].itertuples(index=False))
for x, y, _, z in df_pulsemap_distinct.itertuples(index=False):
    x, y, z = round(x, 2), round(y, 2), round(z, 2)
    assert (x, y, z) in all_dom_pos, (x, y, z)


# %% [markdown]
# # Define utils
def _parallelize_dataframe(df, func, *, n_cores=128):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def _map_df_to_feature_map(df, feature_map):
    return df.apply(
        lambda row: feature_map[row['dom_x'], row['dom_y'], row['dom_z']],
        axis=1, result_type='expand',
    )


def map_df_to_feature_map(df, feature_map, *, n_cores=128):
    return _parallelize_dataframe(
        df,
        partial(_map_df_to_feature_map, feature_map=feature_map),
        n_cores=n_cores
    ).to_numpy()


# %% [markdown]
# # Make icedata transformer (QuantileTransformer)

# Tilt
dom_features = map_df_to_feature_map(df_pulsemap, feature_map)
transformer = QuantileTransformer(output_distribution='normal')
# transformer = RobustScaler()
dom_features_transformed = transformer.fit_transform(dom_features)

with open(f'{EXPORT_DIR}/transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)

# Flat
dom_features_flat = map_df_to_feature_map(df_pulsemap, feature_map_flat)
transformer_flat = QuantileTransformer(output_distribution='normal')
# transformer_flat = RobustScaler()
dom_features_flat_transformed = transformer_flat.fit_transform(dom_features_flat)

with open(f'{EXPORT_DIR}/transformer_flat.pkl', 'wb') as f:
    pickle.dump(transformer_flat, f)


# %% [markdown]
# # Plot inputs and outputs of transformers
def plot_raw_and_transformed_feature_histogram(features, features_transformed):
    NBINSX = 100
    fig = make_subplots(
        rows=len(COLUMNS_ICEMODEL_INTERESTING), cols=2,
        row_titles=[columns_icemodel[i] for i in COLUMNS_ICEMODEL_INTERESTING],
        column_titles=['Raw', 'Transformed'],
    )
    for i, feature in enumerate(COLUMNS_ICEMODEL_INTERESTING):
        fig.add_histogram(x=(features[:, feature]),
                          nbinsx=NBINSX, row=i+1, col=1)
        fig.add_histogram(x=(features_transformed[:, feature]),
                          nbinsx=NBINSX, row=i+1, col=2)

    fig.update_layout(
        title='Feature Distribution',
        width=1000,
        height=1000,
        showlegend=False,
    )
    return fig


# fig = plot_raw_and_transformed_feature_histogram(dom_features, dom_features_transformed)
# fig.show()
# fig.write_html(f'{EXPORT_DIR}/transform.html')

# fig = plot_raw_and_transformed_feature_histogram(dom_features_flat, dom_features_flat_transformed)
# fig.show()
# fig.write_html(f'{EXPORT_DIR}/transform_flat.html')


# %% [markdown]
# # Make DOM icedata lookup table - Transformed
feature_map_transformed = {xyz: transformer.transform([feature]).flatten()
                           for xyz, feature in feature_map.items()}
feature_map_flat_transformed = {xyz: transformer_flat.transform([feature]).flatten()
                                for xyz, feature in feature_map_flat.items()}

with open(f'{EXPORT_DIR}/feature_map.pkl', 'wb') as f:
    pickle.dump(feature_map, f)
with open(f'{EXPORT_DIR}/feature_map_transformed.pkl', 'wb') as f:
    pickle.dump(feature_map_transformed, f)
with open(f'{EXPORT_DIR}/feature_map_flat.pkl', 'wb') as f:
    pickle.dump(feature_map_flat, f)
with open(f'{EXPORT_DIR}/feature_map_flat_transformed.pkl', 'wb') as f:
    pickle.dump(feature_map_flat_transformed, f)


def assert_feature_maps():
    features_transformed = map_df_to_feature_map(df_pulsemap_distinct, feature_map_transformed)
    features_transformed_tf = transformer.transform(
        map_df_to_feature_map(df_pulsemap_distinct, feature_map))
    assert (features_transformed == features_transformed_tf).all()

    features_flat_transformed = map_df_to_feature_map(df_pulsemap_distinct, feature_map_flat_transformed)
    features_flat_transformed_tf = transformer_flat.transform(
        map_df_to_feature_map(df_pulsemap_distinct, feature_map_flat))
    assert (features_flat_transformed == features_flat_transformed_tf).all()


assert_feature_maps()


# %% [markdown]
# # Test transformer on pulsemap
# dom_features = map_df_to_feature_map(df_pulsemap[:100000], feature_map)
# dom_features_transformed = transformer.transform(dom_features)

# fig = plot_raw_and_transformed_feature_histogram(dom_features, dom_features_transformed)
# fig.show()
# fig.write_html(f'{EXPORT_DIR}/transform_example_100000.html')

# dom_features = map_df_to_feature_map(df_pulsemap[:1000000], feature_map)
# dom_features_transformed = transformer.transform(dom_features)

# fig = plot_raw_and_transformed_feature_histogram(dom_features, dom_features_transformed)
# fig.show()


# %% [markdown]
# # Scatter feature plots
def plot_raw_and_transformed_feature_scatter(sws, zs, *features, column_titles):
    # Data for Spice 'truth'/source plot
    features_icemodel = df_icemodel.iloc[31:138, 1:].to_numpy()
    zs_icemodel = -df_icemodel.iloc[31:138, 0]

    fig = make_subplots(
        # rows=1, cols=4,
        rows=len(COLUMNS_ICEMODEL_INTERESTING), cols=len(features)+1,
        row_titles=[columns_icemodel[i] for i in COLUMNS_ICEMODEL_INTERESTING],
        column_titles=column_titles,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.04,
    )
    for i, feature in enumerate(COLUMNS_ICEMODEL_INTERESTING):
        for j, feature_array in enumerate(features):
            fig.add_trace(go.Scatter(
                x=sws, y=zs,
                mode='markers',
                marker=dict(color=feature_array[:, feature], size=4),
                text=feature_array[:, feature],
            ), row=i+1, col=j+1)
            fig.update_xaxes(title_text="Direction SW-225deg", row=i+1, col=j+1)
            if j == 0:
                fig.update_yaxes(title_text="z_global", row=i+1, col=j+1)

        fig.add_trace(go.Scatter(
            x=features_icemodel[:, feature], y=zs_icemodel,
        ), row=i+1, col=len(features)+1)
        fig.update_xaxes(title_text="Feature Value", row=i+1, col=len(features)+1)
        # fig.update_yaxes(title_text="z_global", row=i+1, col=len(features)+1)

    fig.update_layout(
        title='Features',
        width=500 * (len(features)+1),
        height=1600,
        showlegend=False,
    )
    return fig

# %%


def heatplots():
    sws = sw_225(df_pulsemap_distinct[['dom_x', 'dom_y']])
    zs = df_pulsemap_distinct['dom_z_global']

    features = map_df_to_feature_map(df_pulsemap_distinct, feature_map)
    features_transformed = map_df_to_feature_map(df_pulsemap_distinct, feature_map_transformed)
    # features_transformed = transformer.transform(features)
    features_flat = map_df_to_feature_map(df_pulsemap_distinct, feature_map_flat)
    features_flat_transformed = map_df_to_feature_map(df_pulsemap_distinct, feature_map_flat_transformed)
    # features_flat_transformed = transformer_flat.transform(features_flat)

    fig = plot_raw_and_transformed_feature_scatter(
        sws, zs,
        features, features_transformed, features_flat, features_flat_transformed,
        column_titles=['Tilt Raw', 'Tilt Transformed', 'Flat Raw', 'Flat Transformed', 'Spice']
    )
    fig.show(renderer='jpeg')
    fig.write_html(f'{EXPORT_DIR}/feature_2d.html')


def heatplots_for_thesis():
    sws = sw_225(df_pulsemap_distinct[['dom_x', 'dom_y']])
    zs = df_pulsemap_distinct['dom_z_global']

    features = map_df_to_feature_map(df_pulsemap_distinct, feature_map)
    # features_transformed = map_df_to_feature_map(df_pulsemap_distinct, feature_map_transformed)
    # features_transformed = transformer.transform(features)
    # features_flat = map_df_to_feature_map(df_pulsemap_distinct, feature_map_flat)
    # features_flat_transformed = map_df_to_feature_map(df_pulsemap_distinct, feature_map_flat_transformed)
    # features_flat_transformed = transformer_flat.transform(features_flat)

    fig = plot_raw_and_transformed_feature_scatter(
        sws, zs,
        features,  # features_transformed, features_flat, features_flat_transformed,
        column_titles=['Tilted Icedata', 'SPICE']
    )
    fig.show(renderer='jpeg')
    fig.write_image(f'{EXPORT_DIR}/feature_2d.jpg')


heatplots()

# %% [markdown]
# # Compare with Martin


def compare_with_martin():
    sws = sw_225(df_pulsemap_distinct[['dom_x', 'dom_y']])
    zs = df_pulsemap_distinct['dom_z_global']

    features = map_df_to_feature_map(df_pulsemap_distinct, feature_map)

    fig = plot_raw_and_transformed_feature_scatter(
        sws, zs,
        features,
        column_titles=['Tims Features', 'Spice']
    )
    fig.show(renderer='jpg')

    # martin
    x = dict()
    y = dict()
    z = dict()
    for index, row in df_geo.iterrows():
        string_no = row['string_no']
        om_no = row['om_no']

        if string_no not in x:
            x[string_no] = dict()
            y[string_no] = dict()
            z[string_no] = dict()

        x[string_no][om_no] = row['x']
        y[string_no][om_no] = row['y']
        z[string_no][om_no] = row['z']

    def attach_xyz_to_df(df: pd.DataFrame):
        df['x'] = df.apply(lambda row: x[row['string_no']][row['om_no']], axis=1)
        df['y'] = df.apply(lambda row: y[row['string_no']][row['om_no']], axis=1)
        df['z'] = df.apply(lambda row: z[row['string_no']][row['om_no']], axis=1)
        return df

    df_strx = pd.read_csv('../strx', sep=' ', names=['string_no', 'om_no', 'layer_no'])
    attach_xyz_to_df(df_strx)

    sws = sw_225(df_strx[['x', 'y']])
    zs = df_strx['z']

    features = np.array([df_icemodel.iloc[len(df_icemodel)-index, 1:] for index in df_strx['layer_no']])

    fig = plot_raw_and_transformed_feature_scatter(
        sws, zs,
        features,
        column_titles=['Martins Features', 'Spice']
    )
    fig.show(renderer='jpg')


compare_with_martin()


# %%
# PLOT/TEST THIS
# with open('/home/iwsatlas1/timg/remote/icedata/export_old/interp1d_quadratic_A.pkl', 'rb') as f:
#     interp_A = pickle.load(f)
# with open('/home/iwsatlas1/timg/remote/icedata/export_old/interp1d_quadratic_B.pkl', 'rb') as f:
#     interp_B = pickle.load(f)

# with open('/home/iwsatlas1/timg/remote/icedata/export_old/QuantileTransformer_A.pkl', 'rb') as f:
#     tf_A = pickle.load(f)
# with open('/home/iwsatlas1/timg/remote/icedata/export_old/QuantileTransformer_B.pkl', 'rb') as f:
#     tf_B = pickle.load(f)


# def heatplots():
#     sws = sw_225(df_pulsemap_distinct[['dom_x', 'dom_y']])
#     zs = df_pulsemap_distinct['dom_z_global']
#     zs_offset = -zs.to_numpy().reshape(-1, 1)

#     features_A = interp_A(zs_offset)
#     features_A_transformed = tf_A.transform(features_A)
#     features_B = interp_B(zs_offset)
#     features_B_transformed = tf_B.transform(features_B)

#     features = np.column_stack([features_A, features_B, features_A, features_B, features_A, features_B])
#     features_transformed = np.column_stack([features_A_transformed, features_B_transformed,
#                                            features_A_transformed, features_B_transformed, features_A_transformed, features_B_transformed])

#     sws = sw_225(df_pulsemap_distinct[['dom_x', 'dom_y']])
#     zs = df_pulsemap_distinct['dom_z_global']

#     _features = map_df_to_feature_map(df_pulsemap_distinct, feature_map)
#     _features_transformed = map_df_to_feature_map(df_pulsemap_distinct, feature_map_transformed)

#     fig = plot_raw_and_transformed_feature_scatter(
#         sws, zs, _features, _features_transformed, features, features_transformed)
#     fig.show(renderer='jpeg')
#     fig.write_html(f'{EXPORT_DIR}/feature_2d.html')


# heatplots()

# %%
