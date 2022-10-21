import argparse
import sqlite3 as sql
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
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/real_data/data_with_reco/moonL4_segspline_exp13_01_merged_with_time_and_reco_and_new_pulsemap.db",
)
parser.add_argument(
    "-o", "--output", dest="output", type=str, help="the output path [str]"
)
parser.add_argument(
    "-p",
    "--pulsemap",
    dest="pulsemap",
    type=str,
    help="the pulsemap used [str]",
    default="TWSRTHVInIcePulses",
)
args = parser.parse_args()

# dataloading
with sql.connect(args.path_to_db) as con:
    query = """
    SELECT
        dom_time, event_no
    FROM 
        %s;
    """ % (
        args.pulsemap
    )
    sql_data = read_sql(query, con)

pulse_counts = sql_data.groupby("event_no").size()

max_times = sql_data.groupby("event_no").max()
min_times = sql_data.groupby("event_no").min()
event_durations = max_times - min_times

fig, axes = plt.subplots(1, 2, figsize=double)

axes[0].hist(pulse_counts)
axes[0].set_title("Pulse counts")
axes[0].set_yscale("log")
axes[0].set_xlabel("# of pulses")

axes[1].hist(event_durations)
axes[1].set_title("event durations")
axes[1].set_yscale("log")
axes[1].set_xlabel("duration in ns")

fig.tight_layout()
fig.savefig(args.output + "Event_durations_and_Pulse_counts_of_real_data")
