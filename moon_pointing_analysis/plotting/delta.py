import sqlite3 as sql
from plot_params import *

import numpy as np
import pandas as pd
from pandas import read_sql

# bin_number = 50

# data pathing
# indir = "/groups/icecube/qgf305/storage/MoonPointing/Models/inference/Sschindler_data_L4/Merged_database/"
outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/plots/reconstruction"

# dataloading
# azimuth = pd.read_csv(indir + "azimuth_results.csv")
# zenith = pd.read_csv(indir + "zenith_results.csv")

# assign empty containers and plotting parameters
# grid = plt.GridSpec(1,2, wspace=0.3, hspace=.3)
# fig,ax = plt.subplots(1,2,figsize=double)

## plot into containers
# plt.subplot(grid[0, 0])
# plt.title("azimuth prediction")
# mappable = plt.hist2d(
#    azimuth.azimuth_pred, azimuth.azimuth,
#    bins = bin_number,cmap='viridis')
# colorbar(mappable[3])

# plt.subplot(grid[0, 1])
# plt.title("zenith prediction")
# mappable = plt.hist2d(
#    zenith.zenith_pred, zenith.zenith,
#    bins = bin_number,cmap='viridis')
# colorbar(mappable[3])

# save the plot
# plt.savefig(outdir+"AngleResults.png")


db = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/real_data/moonL4_segspline_exp13_01_redo_with_MoonDirection/moonL4_segspline_exp13_01_redo_merged_with_time.db"
with sql.connect(db) as con:
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
        InIceDSTPulses;
    """
    sql_data["time"] = read_sql(query_time, con)


def rad_to_deg(data):
    return (data * 180) / np.pi


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
plt.savefig(outdir + "moon_direction.png")
