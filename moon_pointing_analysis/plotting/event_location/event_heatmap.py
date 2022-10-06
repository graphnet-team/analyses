import sqlite3 as sql
from plot_params import *
from pandas import read_sql
import numpy as np
import itertools
from matplotlib.colors import LogNorm

bins = 25

# data pathing
indir = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/Merged_database.db"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/event_location/test_plot/"

# data contains: charge, dom_time, dom_x, dom_y, dom_z, event_no, pmt_area, rde, width

# dataloading
with sql.connect(indir) as con:
    query = """
    SELECT
        dom_time, dom_x, dom_y, dom_z, event_no
    FROM 
        InIceDSTPulses;
    """
    sql_data = read_sql(query,con)

#hist, xedge, yedge = np.histogram2d(sql_data["dom_x"], sql_data["dom_y"], bins=bins)

#plt.figure(figsize=single)
#X, Y = np.meshgrid(xedge, yedge, sparse=True)
#plt.pcolormesh(X, Y, hist, cmap = 'coolwarm')
#plt.colorbar(label="triggers")
#plt.savefig(outdir + "trigger_density_pmesh.png")

# assign empty containers and plotting parameters
grid = plt.GridSpec(1,3, wspace=0.6, hspace=.3)
fig,ax = plt.subplots(1,3,figsize=triple)

## plot into containers
for i, dom in enumerate(itertools.combinations(["dom_x", "dom_y", "dom_z"], 2)): 
    plt.subplot(grid[0, i])
    # Creating plot
    maxima = max(np.concatenate((sql_data[dom[0]],sql_data[dom[1]]),axis=None))
    mappable = plt.hexbin(
        sql_data[dom[0]], sql_data[dom[1]], gridsize=bins, 
        norm=LogNorm(),
        cmap = 'coolwarm'
        )
    colorbar(mappable)
    plt.xlabel(dom[0]+" position")
    plt.ylabel(dom[1]+" position")
    plt.title(f"trigger density")


plt.savefig(outdir + "trigger_density.png")