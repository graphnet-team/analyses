import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import sqlite3 as sql
import argparse
import numpy as np
import itertools
from matplotlib.colors import LogNorm
from helper_functions.plot_params import *
from pandas import read_sql


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
    "-b",
    "--bins",
    dest="bins",
    type=int,
    help="the number of bins [str]",
    default=25,
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

# dataloading
with sql.connect(args.path_to_db) as con:
    query = """
    SELECT
        dom_time, dom_x, dom_y, dom_z, event_no
    FROM 
        %s;
    """ % (
        args.pulsemap
    )
    sql_data = read_sql(query, con)

# assign empty containers and plotting parameters
grid = plt.GridSpec(1, 3, wspace=0.7, hspace=0.4)
fig, ax = plt.subplots(1, 3, figsize=triple)

## plot trigger density into containers
for i, dom in enumerate(
    itertools.combinations(["dom_x", "dom_y", "dom_z"], 2)
):
    plt.subplot(grid[0, i])
    # Creating plot
    maxima = max(
        np.concatenate((sql_data[dom[0]], sql_data[dom[1]]), axis=None)
    )
    mappable = plt.hexbin(
        sql_data[dom[0]],
        sql_data[dom[1]],
        gridsize=args.bins,
        norm=LogNorm(),
        cmap="coolwarm",
    )
    colorbar(mappable)
    plt.xlabel(dom[0] + " position")
    plt.ylabel(dom[1] + " position")

plt.savefig(args.output + "trigger_density.png")

# decrepit; kept for posterity
# hist, xedge, yedge = np.histogram2d(sql_data["dom_x"], sql_data["dom_y"], bins=bins)

# plt.figure(figsize=single)
# X, Y = np.meshgrid(xedge, yedge, sparse=True)
# plt.pcolormesh(X, Y, hist, cmap = 'coolwarm')
# plt.colorbar(label="triggers")
# plt.savefig(outdir + "trigger_density_pmesh.png")
