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

indir_db = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track.db"
indir_test = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track/osc_next_level3_v2/dynedge_pid_classification3_test/results.csv"
indir_val = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track/osc_next_level3_v2/dynedge_pid_classification3_valid/results.csv"

outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/Selections"


#Load in truth data
with sql.connect(indir_db) as con:
    query = """
    SELECT
        event_no, pid
    FROM 
        truth
    """
    truth_data = read_sql(query,con)

print(truth_data.head(10))
print(truth_data.groupby('pid').count())

test_selection = pd.read_csv(indir_test)
val_selection = pd.read_csv(indir_val)


not_train_selection = pd.concat([test_selection, val_selection], ignore_index=True, sort=False)
train_neutrinos = truth_data[~truth_data['event_no'].isin(not_train_selection['event_no'].to_list())]
train_neutrinos = train_neutrinos['event_no'][train_neutrinos['pid'].isin([-12,12,-14,14,-16,16])]


test_neutrinos = truth_data[truth_data['event_no'].isin(test_selection['event_no'].to_list())]
print('in the test selection there is this many electron neutrinos:')
print(len(test_neutrinos['event_no'][test_neutrinos['pid'].isin([-12,12])]))
print('in the test selection there is this many muon neutrinos:')
print(len(test_neutrinos['event_no'][test_neutrinos['pid'].isin([-14,14])]))
print('in the test selection there is this many tau neutrinos:')
print(len(test_neutrinos['event_no'][test_neutrinos['pid'].isin([-16,16])]))
print('and this many muons:')
print(len(test_neutrinos['event_no'][test_neutrinos['pid'].isin([-13,13])]))
print('and this many noise:')
print(len(test_neutrinos['event_no'][test_neutrinos['pid'].isin([-1,1])]))
test_neutrinos = test_neutrinos['event_no'][test_neutrinos['pid'].isin([-12,12,-14,14,-16,16])]


val_neutrinos = truth_data[truth_data['event_no'].isin(val_selection['event_no'].to_list())]
print('in the val selection there is this many electron neutrinos:')
print(len(val_neutrinos['event_no'][val_neutrinos['pid'].isin([-12,12])]))
print('in the val selection there is this many muon neutrinos:')
print(len(val_neutrinos['event_no'][val_neutrinos['pid'].isin([-14,14])]))
print('in the val selection there is this many tau neutrinos:')
print(len(val_neutrinos['event_no'][val_neutrinos['pid'].isin([-16,16])]))
print('and this many muons:')
print(len(val_neutrinos['event_no'][val_neutrinos['pid'].isin([-13,13])]))
print('and this many noise:')
print(len(val_neutrinos['event_no'][val_neutrinos['pid'].isin([-1,1])]))
val_neutrinos = val_neutrinos['event_no'][val_neutrinos['pid'].isin([-12,12,-14,14,-16,16])]



#train_neutrinos.sort_values('event_no',inplace=True).reset_index()
#val_neutrinos.sort_values('event_no',inplace=True).reset_index()
#test_neutrinos.sort_values('event_no',inplace=True).reset_index()

# train_neutrinos.to_csv(outdir+'neutrinos_train_from_logit_selection.csv')
# val_neutrinos.to_csv(outdir+'neutrinos_val_from_logit_selection.csv')
# test_neutrinos.to_csv(outdir+'neutrinos_test_from_logit_selection.csv')




#RD_event_no.to_csv('dev_lvl3_genie_burnsample_RD_event_numbers.csv',index=False)
#MC_event_no.to_csv('dev_lvl3_genie_burnsample_MC_event_numbers.csv',index=False)