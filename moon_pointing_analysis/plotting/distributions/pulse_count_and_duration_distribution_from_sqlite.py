import sqlite3 as sql
from plot_params import *
import numpy as np
from pandas import cut, read_sql
import pickle as pkl


# data pathing
indir = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/moonL4_segspline_exp13_01_redo_merged_with_time.db"
outdir = "/groups/icecube/peter/workspace/analyses/moon_pointing_analysis/plotting/distributions/test_plot/"

load = True
if load:
    
    filename_1 = outdir + 'pulse_counts.pickle'
    fileObject_1 = open(filename_1, 'rb')
    pulse_counts = pkl.load(fileObject_1)
    fileObject_1.close()

    filename_2 = outdir + 'event_durations.pickle'
    fileObject_2 = open(filename_2, 'rb')
    event_duration = pkl.load(fileObject_2)
    fileObject_2.close()
else:
        
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
        if i%1000 == 0:
            print(i)


    pulse_counts = np.array(pulse_counts)
    event_duration = np.array(event_duration)

    filename = outdir + 'pulse_counts.pickle'
    fileObject = open(filename, 'wb')
    pkl.dump(pulse_counts, fileObject)
    fileObject.close()

    filename = outdir + 'event_durations.pickle'
    fileObject = open(filename, 'wb')
    pkl.dump(event_duration, fileObject)
    fileObject.close()

cut_pulse_count = 1500
cut_event_duration = 0.2*10**6

selection_mask_good = np.array(pulse_counts<cut_pulse_count)*np.array(event_duration<cut_event_duration)
selection_mask_bad = np.array(pulse_counts>cut_pulse_count)*np.array(event_duration>cut_event_duration)

bin_number = 100
bins_pulse = np.linspace(-1,np.max(pulse_counts),bin_number)
bins_duration = np.linspace(-1,np.max(event_duration),bin_number)

fig, axs = plt.subplots(3,2,figsize=(16, 8))

axs[0,0].hist(pulse_counts[selection_mask_good],bins_pulse,color='C0')
axs[0,0].hist(pulse_counts[selection_mask_bad],bins_pulse,color='C1')
axs[0,0].vlines(cut_pulse_count,0,10**5,'black',label=f'cut at {cut_pulse_count}')
axs[0,0].set_title('Whole distribution')
axs[0,0].set_xlabel("# of pulses")
axs[0,0].set_yscale('log')
axs[0,0].legend()

axs[0,1].hist(event_duration[selection_mask_good],bins_duration,color='C0')
axs[0,1].hist(event_duration[selection_mask_bad],bins_duration,color='C1')
axs[0,1].vlines(cut_event_duration,0,10**5,'black',label=f'cut at {cut_event_duration}')
axs[0,1].set_title('Whole distribution')
axs[0,1].set_xlabel("event duration [ns]")
axs[0,1].set_yscale('log')
axs[0,1].legend()

axs[1,0].hist(pulse_counts[selection_mask_good],bin_number,color='C0')
axs[1,0].set_title('Distribution below cut')
axs[1,0].set_xlabel("# of pulses")
axs[1,0].set_yscale('log')

axs[1,1].hist(event_duration[selection_mask_good],bin_number,color='C0')
axs[1,1].set_title('Distribution below cut')
axs[1,1].set_xlabel("duration in ns")
axs[1,1].set_yscale('log')

axs[2,0].hist(pulse_counts[selection_mask_bad],bin_number,color='C1')
axs[2,0].set_title('Distribution above cut')
axs[2,0].set_xlabel("# of pulses")
axs[2,0].set_yscale('log')

axs[2,1].hist(event_duration[selection_mask_bad],bin_number,color='C1')
axs[2,1].set_title('Distribution above cut')
axs[2,1].set_xlabel("duration in ns")
axs[2,1].set_yscale('log')


fig.tight_layout()


fig.savefig(outdir + 'Event_durations_&_Pulse_counts_of_real_data')




