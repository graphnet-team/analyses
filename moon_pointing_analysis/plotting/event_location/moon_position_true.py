import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import sqlite3 as sql
import numpy as np
import pandas as pd
import argparse
from pandas import read_sql
from helper_functions.plot_params import *


parser = argparse.ArgumentParser(
    description="processing i3 files to sqlite3 databases"
)
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_db",
    type=str,
    help="path to database [str]",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="the output path [str]",
    required=True,
)
parser.add_argument(
    "-p",
    "--pulsemap",
    dest="pulsemap",
    type=str,
    help="the pulsemap used [str]",
    required=True,
)
args = parser.parse_args()


with sql.connect(args.path_to_db) as con:
    query = """
    SELECT
        azimuth, zenith
    FROM 
        MoonDirection;
    """
    sql_data = read_sql(query, con)

    query_time = """
    SELECT
        event_time
    FROM 
        %s;
    """ %args.pulsemap
    sql_data["time"] = read_sql(query_time, con)

# define binning
rbins = np.linspace(0, sql_data.zenith.max(), 30)
abins = np.linspace(0, 2 * np.pi, 60)

# calculate histogram
hist, _, _ = np.histogram2d(
    sql_data.azimuth, sql_data.zenith, density=True, bins=(abins, rbins)
)
A, R = np.meshgrid(abins, rbins)

# plot
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(7, 7))

pc = ax.pcolormesh(A, R, hist.T, cmap="magma_r")
fig.colorbar(pc)
plt.grid()
plt.savefig(args.output + "moon_direction.png")
