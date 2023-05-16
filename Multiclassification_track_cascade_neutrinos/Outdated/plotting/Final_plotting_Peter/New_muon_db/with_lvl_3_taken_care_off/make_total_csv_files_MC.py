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
from standard_plotting import set_size

indir_track_cascade_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/track_cascade_New_muon_test_set_inc_truth.csv"
indir_energy_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/energy_New_muon_test_set_inc_truth.csv"
indir_zenith_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/zenith_New_muon_test_set.csv"
indir_azimuth_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/azimuth_New_muon_test_set.csv"
indir_multiclass_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/pid_Multiclass_try_2_on_test_new_muon.csv"
indir_position_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/position_vertex_new_muon.csv"
pulses_strings_doms_MC_indir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/Pulses_doms_strings_times_new_muon_db.csv"
lvl3_variables_MC_indir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/lvl3_variables_new_muon.csv"
track_cascade_MC = pd.read_csv(indir_track_cascade_MC).sort_values('event_no').reset_index(drop = True)
energy_MC = pd.read_csv(indir_energy_MC).sort_values('event_no').reset_index(drop = True)
zenith_MC = pd.read_csv(indir_zenith_MC).sort_values('event_no').reset_index(drop = True)
azimuth_MC = pd.read_csv(indir_azimuth_MC).sort_values('event_no').reset_index(drop = True)
multiclass_MC = pd.read_csv(indir_multiclass_MC).sort_values('event_no').reset_index(drop = True)
position_MC = pd.read_csv(indir_position_MC).sort_values('event_no').reset_index(drop = True)
pulses_strings_doms_MC = pd.read_csv(pulses_strings_doms_MC_indir).sort_values('event_no').reset_index(drop = True)


useful_columns = ['event_no',
       'C2HR6', 'CausalVetoHits', 'CleanedFullTimeLength', 'DCFiducialHits',
       'L3_oscNext_bool', 'NAbove200Hits', 'NchCleaned', 'NoiseEngineNoCharge',
       'RTVetoCutHit', 'STW9000_DTW300Hits', 'UncleanedFullTimeLength',
       'VertexGuessZ', 'VetoFiducialRatioHits']

lvl3_variables_MC = pd.read_csv(lvl3_variables_MC_indir,usecols=useful_columns).sort_values('event_no').reset_index(drop = True)

def to_logit(p):
    eps = 0.0000001
    try:
        if np.isnan(p):
            return
        p = p*(1-2*eps)+eps
        logit = np.log(p/(1-p))
    except ZeroDivisionError as e:
        print(e)
    return logit

multiclass_MC['pid_neutrino_pred_logit'] = pd.Series(multiclass_MC['pid_neutrino_pred']).apply(to_logit)

indir_osc_weight_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/osc_weights_new_muon.csv"
osc_weight_MC = pd.read_csv(indir_osc_weight_MC)
print(len(osc_weight_MC))
print(osc_weight_MC[osc_weight_MC['event_no']==28956376])
osc_weight_MC['osc_weight'][osc_weight_MC['event_no']==28956376] = 0
print(osc_weight_MC[osc_weight_MC['event_no']==28956376])



nr_electron_neutrino_i3_files = 602
nr_muon_neutrino_i3_files = 1518
nr_tau_neutrino_i3_files = 334
nr_muon_i3_files = 1093
nr_noise_i3_files = 401

nr_electron_neutrinos_total = 8118575 + 183333
nr_muon_neutrinos_total = 19923619 + 183333
nr_tau_neutrinos_total = 8702748 + 183333
nr_muon_total = 169737 + 550000
nr_noise_total = 321103 + 550000

nr_electron_neutrinos_test = 8118575
nr_muon_neutrinos_test = 19923619
nr_tau_neutrinos_test = 8702748
nr_muon_test = 169737
nr_noise_test =321103

electron_neutrino_add_weight = 1/(nr_electron_neutrino_i3_files*nr_electron_neutrinos_test/nr_electron_neutrinos_total)
muon_neutrino_add_weight = 1/(nr_muon_neutrino_i3_files*nr_muon_neutrinos_test/nr_muon_neutrinos_total)
tau_neutrino_add_weight = 1/(nr_tau_neutrino_i3_files*nr_tau_neutrinos_test/nr_tau_neutrinos_total)
muon_add_weight = 1/(nr_muon_i3_files*nr_muon_test/nr_muon_total)
noise_add_weight = 1/(nr_noise_i3_files*nr_noise_test/nr_noise_total)
print(electron_neutrino_add_weight,muon_neutrino_add_weight,tau_neutrino_add_weight)

electron_neutrino_event_nos = multiclass_MC['event_no'][multiclass_MC['pid'].isin((-12,12))].to_list()
muon_neutrino_event_nos = multiclass_MC['event_no'][multiclass_MC['pid'].isin((-14,14))].to_list()
tau_neutrino_event_nos = multiclass_MC['event_no'][multiclass_MC['pid'].isin((-16,16))].to_list()
muon_event_nos = multiclass_MC['event_no'][multiclass_MC['pid'].isin((-13,13))].to_list()
noise_event_nos = multiclass_MC['event_no'][multiclass_MC['pid'].isin((-1,1))].to_list()
neutrino_event_nos = multiclass_MC['event_no'][multiclass_MC['pid'].isin((-12,12,-14,14,-16,16))].to_list()

print(osc_weight_MC['osc_weight'][osc_weight_MC['event_no'].isin((electron_neutrino_event_nos))].head(10))
osc_weight_MC['osc_weight'][osc_weight_MC['event_no'].isin((electron_neutrino_event_nos))] *= electron_neutrino_add_weight
print(osc_weight_MC['osc_weight'][osc_weight_MC['event_no'].isin((electron_neutrino_event_nos))].head(10))
osc_weight_MC['osc_weight'][osc_weight_MC['event_no'].isin((muon_neutrino_event_nos))] *= muon_neutrino_add_weight
osc_weight_MC['osc_weight'][osc_weight_MC['event_no'].isin((tau_neutrino_event_nos))] *= tau_neutrino_add_weight
osc_weight_MC['osc_weight'][osc_weight_MC['event_no'].isin((muon_event_nos))] *= muon_add_weight
osc_weight_MC['osc_weight'][osc_weight_MC['event_no'].isin((noise_event_nos))] *= noise_add_weight


print(np.shape(track_cascade_MC))
print(np.shape(energy_MC))
print(np.shape(zenith_MC))
print(np.shape(azimuth_MC))
print(np.shape(multiclass_MC))
print(np.shape(position_MC))
print(np.shape(pulses_strings_doms_MC))
print(np.shape(lvl3_variables_MC))

print(np.shape(osc_weight_MC))


print(track_cascade_MC.head(1))
print(energy_MC.head(1))
print(zenith_MC.head(1))
print(azimuth_MC.head(1))
print(multiclass_MC.head(1))
print(position_MC.head(1))
print(pulses_strings_doms_MC.head(1))
print(lvl3_variables_MC.head(1))

print(osc_weight_MC.head(1))


print(track_cascade_MC.drop(columns=['Unnamed: 0','energy'],inplace=True))
print(energy_MC.drop(columns='Unnamed: 0',inplace=True))
#print(zenith_MC.drop(columns='Unnamed: 0',inplace=True))
#print(azimuth_MC.drop(columns='Unnamed: 0',inplace=True))
print(multiclass_MC.drop(columns='Unnamed: 0',inplace=True))
print(position_MC.drop(columns='Unnamed: 0',inplace=True))
#print(pulses_strings_doms_MC.drop(columns='Unnamed: 0',inplace=True))
#print(lvl3_variables_MC.drop(columns='Unnamed: 0',inplace=True))
#print(retro.drop(columns='Unnamed: 0',inplace=True))
#print(osc_weight_MC.drop(columns='Unnamed: 0',inplace=True))
print(osc_weight_MC.rename(columns={"osc_weight": "total_osc_weight"},inplace=True))


print(track_cascade_MC.head(1))
print(energy_MC.head(1))
print(zenith_MC.head(1))
print(azimuth_MC.head(1))
print(multiclass_MC.head(1))
print(position_MC.head(1))
print(pulses_strings_doms_MC.head(1))
print(lvl3_variables_MC.head(1))

print(osc_weight_MC.head(1))


indir_interaction_type = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/"
MC_interaction_type = pd.read_csv(indir_interaction_type + 'interaction_type_New_muon_test.csv').sort_values('event_no').reset_index(drop = True)

MC_interaction_type.drop(columns='pid',inplace=True)

print(MC_interaction_type.head(1))
print(np.shape(MC_interaction_type))

total_MC = pd.merge(track_cascade_MC.reset_index(drop=True),energy_MC.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)
total_MC = pd.merge(total_MC.reset_index(drop=True),zenith_MC.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)
total_MC = pd.merge(total_MC.reset_index(drop=True),azimuth_MC.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)
total_MC = pd.merge(total_MC.reset_index(drop=True),multiclass_MC.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)
total_MC = pd.merge(total_MC.reset_index(drop=True),position_MC.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)
total_MC = pd.merge(total_MC.reset_index(drop=True),pulses_strings_doms_MC.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)
total_MC = pd.merge(total_MC.reset_index(drop=True),lvl3_variables_MC.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)
total_MC = pd.merge(total_MC.reset_index(drop=True),MC_interaction_type.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)
total_MC = pd.merge(total_MC.reset_index(drop=True),osc_weight_MC.reset_index(drop=True),on='event_no',how='inner').reset_index(drop=True)

print(total_MC.columns)

print(total_MC.head(1))
print(np.shape(total_MC))

outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/track_cascade_sets/"

total_MC.to_csv(outdir + 'Monte_Carlo_all_events_all_variables.csv',index=False)

np.shape(total_MC[total_MC['pid_neutrino_pred_logit']>1])

small_MC = total_MC[total_MC['pid_neutrino_pred_logit']>1]
small_MC.to_csv(outdir + 'Monte_Carlo_neutrino_prob_logit_above_1_all_variables.csv',index=False)

small_small_MC = total_MC[total_MC['pid_neutrino_pred_logit']>12]
small_small_MC.to_csv(outdir + 'Monte_Carlo_neutrino_prob_logit_above_12_all_variables.csv',index=False)
print(np.shape(small_small_MC))

print('all finished')