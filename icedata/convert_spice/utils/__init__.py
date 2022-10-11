from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd


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
