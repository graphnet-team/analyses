"""mostly an example script"""
import sqlite3 as sql
import argparse
from pandas import read_sql
from plot_params import *

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
        charge, dom_time, dom_x, dom_y, dom_z, event_no, pmt_area, rde, width
    FROM 
        %s;
    """ % (
        args.pulsemap
    )
    sql_data = read_sql(query, con)

plt.figure()
plt.hist(sql_data["charge"], bins=10)
plt.yscale("log")
plt.title("input data: Charge")
plt.legend()
plt.savefig(args.output + "L2_2018_1.png")
