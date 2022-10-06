import sqlite3 as sql
from ..plot_params import *

from pandas import read_sql

# data pathing
indir = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/Merged_database.db"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/distributions/test_plot"

# dataloading
with sql.connect(db) as con:
    query = """
    SELECT
        charge, dom_time, dom_x, dom_y, dom_z, event_no, pmt_area, rde, width
    FROM 
        InIceDSTPulses;
    """
    sql_data = read_sql(query,con)

# assign empty containers and plotting parameters
grid = plt.GridSpec(1,2, wspace=0.3, hspace=.3)
fig,ax = plt.subplots(single)

## plot into containers
plt.figure()
plt.hist(sql_data["charge"], bins = 10)
plt.yscale('log')
plt.title("input data: Charge")
plt.legend()
plt.savefig(outdir + "L2_2018_1.png")