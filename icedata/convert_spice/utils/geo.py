from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

root = Path('../ice-bfr-v2-x5-rde')

df = pd.read_csv(root.joinpath('geo-f2k'), sep='\t',
                 names=['dom_id', 'mainboard_id', 'x', 'y', 'z', 'string_no', 'om_no'])

x = dict()
y = dict()
z = dict()
for index, row in df.iterrows():
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


if __name__ == '__main__':
    print(df)
    del df['x']
    del df['y']
    del df['z']
    print(df)
    print(f'{attach_xyz_to_df(df) = }')
