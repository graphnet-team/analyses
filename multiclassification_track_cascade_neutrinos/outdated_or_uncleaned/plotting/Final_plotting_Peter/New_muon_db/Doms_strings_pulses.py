import sqlite3 as sql

import numpy as np
import pandas as pd
from pandas import cut, read_sql
import pickle as pkl
from random import choices
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.optimize as optimize

outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/"
indir_db_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"

with sql.connect(indir_db_MC) as con:
        query = f"""
        SELECT
            event_no
        FROM 
            truth
        """
        MC_event_no = read_sql(query,con)


Npulses_MC = []
Ndoms_MC = []
Nstrings_MC = []
#event_no_MC = []
event_times_MC = []
print('starting now')
#Load in truth data
for i in range(len(MC_event_no)):
    if i%1000 == 0:
        print(i, flush=True)
    with sql.connect(indir_db_MC) as con:
        query = f"""
        SELECT
            event_time, dom_x, dom_y, dom_z
        FROM 
            SplitInIcePulses
        WHERE
            event_no = {MC_event_no['event_no'][i]}
        """
        MC_extra_data = read_sql(query,con)
    #event_no_MC.append(MC_event_no['event_no'][i])
    Npulses_MC.append(len(MC_extra_data))
    Ndoms_MC.append(MC_extra_data.groupby(['dom_x', 'dom_y','dom_z']).ngroups)
    Nstrings_MC.append(MC_extra_data.groupby(['dom_x', 'dom_y']).ngroups)
    event_times_MC.append(MC_extra_data['event_time'][0])
print('MC done')

MC_final = MC_event_no
MC_final['N_pulses'] = Npulses_MC
MC_final['N_string'] = Nstrings_MC
MC_final['N_doms'] = Ndoms_MC
MC_final['First_dom_time'] = event_times_MC

MC_final.to_csv(outdir + 'Pulses_doms_strings_times_new_muon_db.csv',index=False)
print('MC saved')

outdir_RD = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Burnsample/"
indir_db_RD = "/groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"

with sql.connect(indir_db_RD) as con:
        query = f"""
        SELECT
            event_no
        FROM 
            truth
        """
        RD_event_no = read_sql(query,con)


Npulses_RD = []
Ndoms_RD = []
Nstrings_RD = []
#event_no_RD = []
first_dom_time_RD = []
for i in range(len(RD_event_no)):
    if i%1000 == 0:
        print(i, flush=True)
    #Load in truth data
    with sql.connect(indir_db_RD) as con:
        query = f"""
        SELECT
            dom_time, dom_x, dom_y, dom_z
        FROM 
            SplitInIcePulses
        WHERE
            event_no = {RD_event_no['event_no'][i]}
        """
        RD_extra_data = read_sql(query,con)
    #event_no_RD.append(csv_cascade_RD['event_no'][i])
    Npulses_RD.append(len(RD_extra_data))
    Ndoms_RD.append(RD_extra_data.groupby(['dom_x', 'dom_y','dom_z']).ngroups)
    Nstrings_RD.append(RD_extra_data.groupby(['dom_x', 'dom_y']).ngroups)
    first_dom_time_RD.append(RD_extra_data['dom_time'][0])
print('RD Done')



RD_final = RD_event_no
RD_final['N_pulses'] = Npulses_RD
RD_final['N_string'] = Nstrings_RD
RD_final['N_doms'] = Ndoms_RD
RD_final['First_dom_time'] = first_dom_time_RD

RD_final.to_csv(outdir_RD + 'Pulses_doms_strings_times_burnsample.csv',index=False)

print('RD saved')