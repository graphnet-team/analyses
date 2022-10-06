import sqlite3 as sql
from plot_params import *

import pandas as pd
from pandas import read_sql

#bin_number = 50

# data pathing
#indir = "/groups/icecube/qgf305/storage/MoonPointing/Models/inference/Sschindler_data_L4/Merged_database/"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/reconstruction/test_plot/"

# dataloading
#azimuth = pd.read_csv(indir + "azimuth_results.csv")
#zenith = pd.read_csv(indir + "zenith_results.csv")

# assign empty containers and plotting parameters
#grid = plt.GridSpec(1,2, wspace=0.3, hspace=.3)
#fig,ax = plt.subplots(1,2,figsize=double)

## plot into containers
#plt.subplot(grid[0, 0])
#plt.title("azimuth prediction")
#mappable = plt.hist2d(
#    azimuth.azimuth_pred, azimuth.azimuth, 
#    bins = bin_number,cmap='viridis')
#colorbar(mappable[3])

#plt.subplot(grid[0, 1])
#plt.title("zenith prediction")
#mappable = plt.hist2d(
#    zenith.zenith_pred, zenith.zenith, 
#    bins = bin_number,cmap='viridis')
#colorbar(mappable[3])

# save the plot
#plt.savefig(outdir+"AngleResults.png")


db = "/groups/icecube/qgf305/storage/databases/moonL4_segspline_exp13_01_redo_with_MoonDirection.db"
with sql.connect(db) as con:
    query = """
    SELECT
        azimuth, zenith
    FROM 
        MoonDirection;
    """
    sql_data = read_sql(query,con)

plt.figure()
plt.hist(sql_data["zenith"], bins = 10)
plt.yscale('log')
plt.title("Moon zenith location")
plt.legend()
plt.savefig(outdir + "moon_zenith.png")

plt.figure()
plt.hist(sql_data["azimuth"], bins = 10)
plt.yscale('log')
plt.title("Moon azimuth location")
plt.legend()
plt.savefig(outdir + "moon_azimuth.png")
