import sqlite3 as sql
from plot_params import *

import pandas as pd
from pandas import read_sql

bin_number = 50

# data pathing
indir = "/groups/icecube/qgf305/storage/MoonPointing/Models/inference/Sschindler_data_L4/Merged_database/"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/reconstruction/test_plot/"

# dataloading
azimuth = pd.read_csv(indir + "azimuth_results.csv")
zenith = pd.read_csv(indir + "zenith_results.csv")

# assign empty containers and plotting parameters
grid = plt.GridSpec(1,2, wspace=0.3, hspace=.3)
fig,ax = plt.subplots(1,2,figsize=double)

## plot into containers
plt.subplot(grid[0, 0])
plt.title("azimuth prediction")
mappable = plt.hist2d(
    azimuth.azimuth_pred, azimuth.azimuth, 
    bins = bin_number,cmap='viridis')
colorbar(mappable[3])

plt.subplot(grid[0, 1])
plt.title("zenith prediction")
mappable = plt.hist2d(
    zenith.zenith_pred, zenith.zenith, 
    bins = bin_number,cmap='viridis')
colorbar(mappable[3])

# save the plot
plt.savefig(outdir+"AngleResults.png")


db = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/Merged_database.db"
with sql.connect(db) as con:
    query = """
    SELECT
        charge, dom_time, dom_x, dom_y, dom_z, event_no, pmt_area, rde, width
    FROM 
        InIceDSTPulses;
    """
    sql_data = read_sql(query,con)

plt.figure()
plt.hist(sql_data["charge"], bins = 10)
plt.yscale('log')
plt.title("input data: Charge")
plt.legend()
plt.savefig(outdir + "L2_2018_1.png")

