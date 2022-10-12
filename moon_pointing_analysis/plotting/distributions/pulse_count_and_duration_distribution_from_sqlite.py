import sqlite3 as sql
from plot_params import *
import numpy as np
from pandas import read_sql

# data pathing
indir = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/moonL4_segspline_exp13_01_redo_merged_with_time.db"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/distributions/test_plot"

# dataloading
with sql.connect(indir) as con:
    query = """
    SELECT
        charge, dom_time, dom_x, dom_y, dom_z, event_no, pmt_area, rde, width
    FROM 
        InIceDSTPulses;
    """
    sql_data = read_sql(query,con)

pulse_counts = []
event_duration = []
i = 0
for nr in sql_data['event_no'].unique():
    i+=1
    pulse_counts.append(len(sql_data[sql_data['event_no']==nr]))
    max_time = np.max(sql_data[sql_data['event_no']==nr]['dom_time'])
    min_time = np.min(sql_data[sql_data['event_no']==nr]['dom_time'])
    event_duration.append(max_time-min_time)
    if i%5000 == 0:
        print(i)
    
fig, axs = plt.subplots(1,2,figsize=(16, 8))

axs[0,0].hist(pulse_counts)
axs[0,0].set_title('Pulse counts')
axs[0,0].set_xlabel("# of pulses")

axs[0,1].hist(event_duration)
axs[0,1].set_title('event durations')
axs[0,1].set_xlabel("duration in ns")

fig.tight_layout()
fig.savefig(outdir + 'Event_durations_&_Pulse_counts_of_real_data')
