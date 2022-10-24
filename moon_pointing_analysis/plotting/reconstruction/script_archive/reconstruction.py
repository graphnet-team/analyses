import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import sqlite3 as sql
import pandas as pd
import argparse
from pandas import read_sql
from helper_functions.plot_params import *

parser = argparse.ArgumentParser(
    description="plotting the predicted zenith and azimuth vs truth."
)
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_csv",
    type=str,
    help="path to database [str]",
    default="/groups/icecube/peter/storage/MoonPointing/Models/Leon_Muon_data_MC/last_one_lvl3MC/dynedge_zenith_Leon_muon_data_MC/results.csv",
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="the output path [str]",
    default="/groups/icecube/qgf305/workspace/analyses/moon_pointing_analysis/plotting/reconstruction/test_plot",
)
parser.add_argument(
    "-b",
    "--bins",
    dest="bins",
    type=int,
    help="the number of bins [str]",
    default=25,
)
args = parser.parse_args()

# dataloading
azimuth = pd.read_csv(indir + "azimuth_results.csv")
zenith = pd.read_csv(indir + "zenith_results.csv")

# assign empty containers and plotting parameters
grid = plt.GridSpec(1, 2, wspace=0.3, hspace=0.3)
fig, ax = plt.subplots(1, 2, figsize=double)

## plot into containers
plt.subplot(grid[0, 0])
plt.title("azimuth prediction")
mappable = plt.hist2d(
    azimuth.azimuth_pred, azimuth.azimuth, bins=bin_number, cmap="viridis"
)
colorbar(mappable[3])

plt.subplot(grid[0, 1])
plt.title("zenith prediction")
mappable = plt.hist2d(
    zenith.zenith_pred, zenith.zenith, bins=bin_number, cmap="viridis"
)
colorbar(mappable[3])

# save the plot
plt.savefig(outdir + "AngleResults.png")
