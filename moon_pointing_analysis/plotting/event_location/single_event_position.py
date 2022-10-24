import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import sqlite3 as sql
import argparse
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
        charge, dom_time, dom_x, dom_y, dom_z, event_no
    FROM 
        %s;
    """ % (
        args.pulsemap
    )
    sql_data = read_sql(query, con)

event_numbers = sql_data["event_no"].unique()
specific_event = sql_data[sql_data["event_no"] == event_numbers[0]]

fig = plt.figure()
ax = plt.axes(projection="3d")

# Creating plot
dom = ax.scatter3D(
    specific_event["dom_x"],
    specific_event["dom_y"],
    specific_event["dom_z"],
    c=specific_event["dom_time"],
    cmap="coolwarm",
    s=25, alpha=1
)  # TODO; s should vary in size like the official plots
colorbar(dom)
ax.set_xlabel("x position")
ax.set_ylabel("x position")
ax.set_zlabel("Z Label")
plt.title(
    f"simple 3D scatter plot of dom positions for event #{event_numbers[0]}"
)
plt.savefig(args.output + "single_event_position.png")
