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

indir_RD = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/inference/pid_Burnsample_RD_Full_db.csv"
indir_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/trained_models/osc_next_level3_v2/dynedge_pid_classification3_test/results.csv"
outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/comparison_"

results_RD = pd.read_csv(indir_RD).sort_values('event_no').reset_index(drop = True)
results_MC = pd.read_csv(indir_MC).sort_values('event_no').reset_index(drop = True)

bins_to_use = np.linspace(0,1,101)
bins_middle = (bins_to_use[1:]+bins_to_use[:-1])/2

print(bins_to_use)
print(bins_middle)

pid_transform = {1:0,12:2,13:1,14:2,16:2}

truth_MC = []

for i in range(len(results_MC)):# range(len(results)):
    truth_MC.append(pid_transform[abs(results_MC['pid'].values[i])])

mask_noise = [True if truth_MC[i] ==0 else False for i in range(len(truth_MC))]
mask_muon = [True if truth_MC[i] ==1 else False for i in range(len(truth_MC))]
mask_neutrino = [True if truth_MC[i] ==2 else False for i in range(len(truth_MC))]

fig, axs = plt.subplots(4,1,sharex=False,figsize=(8, 32))

counts_neutrino, _,_ = axs[0].hist(results_MC['pid_neutrino_pred'].values[mask_neutrino],bins=bins_to_use,label='MC Neutrinos',density=True)
counts_muon, _,_ = axs[1].hist(results_MC['pid_neutrino_pred'].values[mask_muon],bins=bins_to_use,label='MC muons',density=True)
counts_noise, _,_ = axs[2].hist(results_MC['pid_neutrino_pred'].values[mask_noise],bins=bins_to_use,label='MC Noise',density=True)
counts_RD, _,_ = axs[3].hist(results_RD['pid_neutrino_pred'].values,bins=bins_to_use,label='RD')

print(counts_noise[:5])
print(counts_RD[:5])

axs[0].set_ylabel('Density')
axs[1].set_ylabel('Density')
axs[2].set_ylabel('Density')
axs[3].set_ylabel('Count')

axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')
axs[3].set_yscale('log')

axs[0].set_xlabel('Neutrino probability')
axs[1].set_xlabel('Neutrino probability')
axs[2].set_xlabel('Neutrino probability')
axs[3].set_xlabel('Neutrino probability')

axs[0].set_title('MC Neutrino')
axs[1].set_title('MC Muons')
axs[2].set_title('MC Noise')
axs[3].set_title('RD')

fig.tight_layout()

fig.savefig(outdir + 'Neutrino_prob_histograms.png')

def chi_square(N_scalers):
    sum = 0
    for i in range(len(bins_to_use)-1):
        sum+= (counts_RD[i] - N_scalers[0]*counts_noise[i] - N_scalers[1]*counts_muon[i] - N_scalers[2]*counts_neutrino[i])**2/counts_RD[i]
    return sum

N_0 = [50000,50000,50000]
res = optimize.minimize(chi_square,N_0,bounds = ((0, None),(0, None), (0, None)))
print(res.x)
print(res.success)

N_test = [500000,500000,50000]
fig, axs = plt.subplots(figsize=(8, 8))

counts_noise_fit = counts_noise*res.x[0]#*N_test[0]
counts_muon_fit = counts_muon*res.x[1]#*N_test[1]#
counts_neutrino_fit = counts_neutrino*res.x[2]#*N_test[2]#

print('noise:',counts_noise_fit[:10])
print('muon:',counts_muon_fit[:10])
print('neutrino:',counts_neutrino_fit[:10])

axs.stairs(counts_neutrino_fit, bins_to_use,baseline=(counts_noise_fit+counts_muon_fit),label='Scaled Neutrinos',fill=True,color='green')
axs.stairs(counts_muon_fit, bins_to_use,baseline=counts_noise_fit,label='Scaled Muons',fill=True,color='orange')
axs.stairs(counts_noise_fit, bins_to_use,label='Scaled Noise',fill=True,color='blue')


axs.plot(bins_middle,counts_RD,'o',label='Real data')
axs.set_xlabel('Neutrino probability')

axs.set_ylabel('Count')
axs.set_yscale('log')
axs.legend(loc='upper right')

fig.tight_layout()

fig.savefig(outdir + 'Scaled_prob_histograms.png')