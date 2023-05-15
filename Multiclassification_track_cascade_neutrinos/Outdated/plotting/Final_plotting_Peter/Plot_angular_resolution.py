import sqlite3 as sql

import numpy as np
import pandas as pd
from pandas import cut, read_sql
import pickle as pkl
from random import choices
import matplotlib.pyplot as plt

fs=45
indir_track_cascade_test = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/trained_models/osc_next_level3_v2/dynedge_track_mu_Track_cascade_MP_data_SplitInIcePulses_on_equal_track_cascade_neutrinos_test/results.csv"
track_cascade_test = pd.read_csv(indir_track_cascade_test).sort_values('event_no').reset_index(drop = True)
track_selection_test = track_cascade_test['event_no'][track_cascade_test['track_mu_pred'] > 0.9]

indir_track_cascade_val = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/trained_models/osc_next_level3_v2/dynedge_track_mu_Track_cascade_MP_data_SplitInIcePulses_on_equal_track_cascade_neutrinos_validation/results.csv"
track_cascade_val = pd.read_csv(indir_track_cascade_val).sort_values('event_no').reset_index(drop = True)
track_selection_val = track_cascade_val['event_no'][track_cascade_val['track_mu_pred'] > 0.9]


def angular_resolution(angle,bins,predicted_database,degrees=True,bootstrap=True,bootstrap_nr_samples=100):
    energy = predicted_database['energy']
    
    if angle=='zenith':
        angle_pred = predicted_database['zenith_pred']
        angle_true = predicted_database['zenith']

    elif angle=='azimuth':
        angle_pred = predicted_database['azimuth_pred']
        angle_true = predicted_database['azimuth']


    bin_energy_middle = (bins[:-1] + bins[1:])/2
    bin_size = []
    bin_angle_resolution_pred = []
    bootstrap_std = []
    for i in range(len(bins)-1):
        low = bins[i]
        high = bins[i+1]

        mask =  (energy > low) & (energy < high)

        bin_size.append(np.sum(mask))
        
        
        bin_angle_residual_pred = (angle_pred[mask]-angle_true[mask])
        
        if angle=="azimuth":
            bin_angle_residual_pred[bin_angle_residual_pred<-np.pi] = 2*np.pi + bin_angle_residual_pred[bin_angle_residual_pred<-np.pi]
            bin_angle_residual_pred[bin_angle_residual_pred>np.pi] = -2*np.pi + bin_angle_residual_pred[bin_angle_residual_pred>np.pi]
        if len(bin_angle_residual_pred) !=0:    
            bin_angle_resolution_pred.append( (np.percentile(bin_angle_residual_pred,84) - np.percentile(bin_angle_residual_pred,16))/2 )
        else:
            bin_angle_resolution_pred.append(0)

        bootstrap_list = []
        for j in range(bootstrap_nr_samples):
            if len(bin_angle_residual_pred) != 0:
                residuals_sample = choices(np.array(bin_angle_residual_pred),k=len(bin_angle_residual_pred))
            else: 
                residuals_sample = []
            if len(residuals_sample) != 0:
                resolution_sample = (np.percentile(residuals_sample,84) - np.percentile(residuals_sample,16))/2
            else:
                print('zero')
                resolution_sample = []
            bootstrap_list.append(resolution_sample)
        bootstrap_std.append(np.std(bootstrap_list))


    if degrees:
        bin_angle_resolution_pred = np.array(bin_angle_resolution_pred)*180/np.pi
        bootstrap_std = np.array(bootstrap_std)*180/np.pi
        #bootstrap_std_2 = np.array(bootstrap_std_2)*180/np.pi

    if bootstrap==True:
        return energy, bin_energy_middle, bin_size, bin_angle_resolution_pred, bootstrap_std
    else:
        return energy, bin_energy_middle, bin_size, bin_angle_resolution_pred 



bootstrap_bool = True

print('running')
# data pathing
outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/plotting/"

#Paths to prediction csv files, append as many as you like
prediction_indirs_zenith_test = ["/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/trained_models/archieve/MP_data_zenith_1_mill_test_set/results.csv"]
prediction_indirs_zenith_val = ["/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/trained_models/archieve/MP_data_zenith_1_mill_validation_set/results.csv"]
prediction_indirs_azimuth_test = ["/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/trained_models/osc_next_level3_v2/MP_data_azimuth_test_1_mill_attempt2_test_set_equal_track_cascade/results.csv"]
prediction_indirs_azimuth_val = ["/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/trained_models/osc_next_level3_v2/MP_data_azimuth_test_1_mill_attempt2_validation_set_equal_track_cascade/results.csv"]

#Changes paths to databases
prediction_database_zenith_test = [pd.read_csv(dir).sort_values('event_no').reset_index(drop = True) for dir in prediction_indirs_zenith_test]
prediction_database_zenith_val = [pd.read_csv(dir).sort_values('event_no').reset_index(drop = True) for dir in prediction_indirs_zenith_val]
prediction_database_azimuth_test = [pd.read_csv(dir).sort_values('event_no').reset_index(drop = True) for dir in prediction_indirs_azimuth_test]
prediction_database_azimuth_val = [pd.read_csv(dir).sort_values('event_no').reset_index(drop = True) for dir in prediction_indirs_azimuth_val]
#prediction_database_zenith_test[0] = prediction_database_zenith_test[0][prediction_database_zenith_test[0]['event_no'].isin(track_selection_test)]
#prediction_database_zenith_val[0] = prediction_database_zenith_val[0][prediction_database_zenith_val[0]['event_no'].isin(track_selection_val)]
#prediction_database_azimuth_test[0] = prediction_database_azimuth_test[0][prediction_database_azimuth_test[0]['event_no'].isin(track_selection_test)]
#prediction_database_azimuth_val[0] = prediction_database_azimuth_val[0][prediction_database_azimuth_val[0]['event_no'].isin(track_selection_val)]

#Labels of the databases
labels = ['baseline']#,'LR=1e-02']

#Colors of the databases for plot
colors = ['C1','C2','C3','C4','C5'] 


#Select energy bins
overall_energy = prediction_database_zenith_test[0]['energy']
plt.figure()
plt.plot(overall_energy)
plt.show()
energy_bins = 10**np.arange(np.min(np.log10(overall_energy)),np.max(np.log10(overall_energy)), 0.1)
#energy_bins = 10**np.arange(1,np.max(np.log10(overall_energy)), 0.1)

#Defines the lists to be used
energies_test,energies_val, bins_energy_middle, bin_sizes_test,bin_sizes_val, bin_zenith_resolutions_test, bin_zenith_resolutions_val, bin_azimuth_resolutions_test, bin_azimuth_resolutions_val = [],[],[],[],[],[],[],[],[]
bin_zenith_resolutions_bootstrap_std_test, bin_zenith_resolutions_bootstrap_std_val, bin_azimuth_resolutions_bootstrap_std_test, bin_azimuth_resolutions_bootstrap_std_val = [],[],[],[]

#Get resolutions for all databases in each bin
for i in range(len(prediction_database_zenith_test)):
    energy_test, bin_energy_middle_test, bin_size_test,  bin_zenith_resolution_pred_test, bootstrap_zenith_std_pred_test = angular_resolution('zenith',energy_bins,prediction_database_zenith_test[i] ,degrees=True,bootstrap=bootstrap_bool)
    energy_val, bin_energy_middle_val, bin_size_val,  bin_zenith_resolution_pred_val,bootstrap_zenith_std_pred_val  = angular_resolution('zenith',energy_bins,prediction_database_zenith_val[i],degrees=True,bootstrap=bootstrap_bool)
    energies_test.append(energy_test)
    energies_val.append(energy_val)
    bins_energy_middle.append(bin_energy_middle_test)
    bin_sizes_test.append(bin_size_test)
    bin_sizes_val.append(bin_size_val)
    bin_zenith_resolutions_test.append(bin_zenith_resolution_pred_test)
    bin_zenith_resolutions_val.append(bin_zenith_resolution_pred_val)
    bin_zenith_resolutions_bootstrap_std_test.append(bootstrap_zenith_std_pred_test)
    bin_zenith_resolutions_bootstrap_std_val.append(bootstrap_zenith_std_pred_val)

for i in range(len(prediction_database_azimuth_test)):
    _, _, _, bin_azimuth_resolution_pred_test,bootstrap_azimuth_std_pred_test = angular_resolution('azimuth',energy_bins,prediction_database_azimuth_test[i],degrees=True,bootstrap=bootstrap_bool)
    _, _, _, bin_azimuth_resolution_pred_val,bootstrap_azimuth_std_pred_val = angular_resolution('azimuth',energy_bins,prediction_database_azimuth_val[i],degrees=True,bootstrap=bootstrap_bool)
    
    bin_azimuth_resolutions_test.append(bin_azimuth_resolution_pred_test)
    bin_azimuth_resolutions_val.append(bin_azimuth_resolution_pred_val)
    bin_azimuth_resolutions_bootstrap_std_test.append(bootstrap_azimuth_std_pred_test)
    bin_azimuth_resolutions_bootstrap_std_val.append(bootstrap_azimuth_std_pred_val)



#Zenith updated versions vs baseline 
fig, axs = plt.subplots(1,1,figsize=(18, 18))
axs2 = axs.twinx()
axs2.hist(energies_test[0],energy_bins,label = 'Number of test events',color='C0',alpha=0.2)

for i in range(len(prediction_database_zenith_test)):
    res = bin_zenith_resolutions_test[i]
    std = bin_zenith_resolutions_bootstrap_std_test[i]

    axs.plot(bins_energy_middle[0],res,'--',label = labels[i],color=colors[i])
    axs.fill_between(bins_energy_middle[0],res-std,res+std,color=colors[i],alpha=0.3)


fs2 = 35
axs.set_ylabel('Zenith Resolution [deg.]',fontsize=fs)
axs.set_xlabel("Energy [GeV]",fontsize=fs)
axs2.set_ylabel('Number of Events',fontsize=fs)
axs2.set_yscale('log')
axs.set_xscale('log')
axs.legend(loc ='upper right',fontsize=fs)
axs2.legend(loc ='upper center',fontsize=fs)
axs.tick_params(axis='x', labelsize=fs2)
axs.tick_params(axis='y', labelsize=fs2)
axs2.tick_params(axis='y', labelsize=fs2)
axs.tick_params(axis='both', which='major', pad=15)
axs.set_ylim(ymin=0)
fig.tight_layout()
plt.tight_layout()
fig.savefig(outdir + 'Zenith_resolution_all')



#Azimuth updated versions vs baseline 
fig, axs = plt.subplots(1,1,figsize=(18, 18))
axs2 = axs.twinx()
axs2.hist(energies_test[0],energy_bins,label = 'Number of test events',color='C0',alpha=0.2)

for i in range(len(prediction_database_azimuth_test)):
    res = bin_azimuth_resolutions_test[i]
    std = bin_azimuth_resolutions_bootstrap_std_test[i]

    axs.plot(bins_energy_middle[0],res,'--',label = labels[i],color=colors[i])
    axs.fill_between(bins_energy_middle[0],res-std,res+std,color=colors[i],alpha=0.3)

    
axs.set_ylabel('Azimuth Resolution [deg.]',fontsize=fs)

axs.set_xlabel("Energy [GeV]",fontsize=fs)
axs2.set_ylabel('Number of Events',fontsize=fs)
axs2.set_yscale('log')
axs.set_xscale('log')
axs.legend(loc ='upper right',fontsize=fs)
axs.tick_params(axis='x', labelsize=fs2)
axs.tick_params(axis='y', labelsize=fs2)
axs2.tick_params(axis='y', labelsize=fs2)
axs.tick_params(axis='both', which='major', pad=15)
axs.set_ylim(ymin=0)
fig.tight_layout()
plt.tight_layout()

fig.savefig(outdir + 'Azimuth_resolution_all')
print('finished plotting')
print('all done')