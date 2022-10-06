import sqlite3 as sql
from plot_params import *
from pandas import read_sql

# data pathing
indir = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/Merged_database.db"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/event_location/test_plot/"

# dataloading
with sql.connect(indir) as con:
    query = """
    SELECT
        charge, dom_time, dom_x, dom_y, dom_z, event_no
    FROM 
        InIceDSTPulses;
    """
    sql_data = read_sql(query,con)

event_numbers = sql_data["event_no"].unique()
specific_event = sql_data[sql_data["event_no"] == event_numbers[0]]

fig = plt.figure()
ax = plt.axes(projection ="3d")
 
# Creating plot
dom = ax.scatter3D(
    specific_event["dom_x"], specific_event["dom_y"], specific_event["dom_z"],
    c = specific_event["dom_time"], cmap = 'coolwarm', s = 20)
fig.colorbar(dom, ax=ax)
ax.set_xlabel("x position")
ax.set_ylabel("x position")
ax.set_zlabel('Z Label')
plt.title(f"simple 3D scatter plot of dom positions for event #{event_numbers[0]}")
plt.savefig(outdir + "scatter.png")