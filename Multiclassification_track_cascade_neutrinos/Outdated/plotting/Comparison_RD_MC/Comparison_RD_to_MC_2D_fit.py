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

indir_RD = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Inference/pid_Leon_RD_results_new_model.csv"
indir_MC = "/groups/icecube/peter/storage/Multiclassification/Real/last_one_lvl3MC/dynedge_pid_Real_run_21.5_mill_equal_frac_second_/results.csv"
outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/2d_comparison_"

results_RD = pd.read_csv(indir_RD).sort_values('event_no').reset_index(drop = True)
results_MC = pd.read_csv(indir_MC).sort_values('event_no').reset_index(drop = True)

bins_to_use = np.linspace(0,1,101)
bins_middle = (bins_to_use[1:]+bins_to_use[:-1])/2

#print(bins_to_use)
#print(bins_middle)

pid_transform = {1:0,12:2,13:1,14:2,16:2}

truth_MC = []

for i in range(len(results_MC)):# range(len(results)):
    truth_MC.append(pid_transform[abs(results_MC['pid'].values[i])])

mask_noise = [True if truth_MC[i] ==0 else False for i in range(len(truth_MC))]
mask_muon = [True if truth_MC[i] ==1 else False for i in range(len(truth_MC))]
mask_neutrino = [True if truth_MC[i] ==2 else False for i in range(len(truth_MC))]

fig, axs = plt.subplots(4,1,sharex=False,figsize=(8, 32))

counts_noise,_,_,_ = axs[0].hist2d(results_MC['pid_noise_pred'].values[mask_noise],results_MC['pid_neutrino_pred'].values[mask_noise],bins_to_use,norm=colors.LogNorm())
counts_muon,_,_,_ = axs[1].hist2d(results_MC['pid_noise_pred'].values[mask_muon],results_MC['pid_neutrino_pred'].values[mask_muon],bins_to_use,norm=colors.LogNorm())
counts_neutrino,_,_,_ = axs[2].hist2d(results_MC['pid_noise_pred'].values[mask_neutrino],results_MC['pid_neutrino_pred'].values[mask_neutrino],bins_to_use,norm=colors.LogNorm())
counts_RD,_,_,_ = axs[2].hist2d(results_RD['pid_noise_pred'].values,results_RD['pid_neutrino_pred'].values,bins_to_use,norm=colors.LogNorm())


def chi_square(N_scalers):
    sum = 0
    for i in range(len(bins_to_use)-1):
        for j in range(len(bins_to_use)-1):
            if counts_RD[i,j] > 0:
                sum+= (counts_RD[i,j] - N_scalers[0]*counts_noise[i,j] - N_scalers[1]*counts_muon[i,j] - N_scalers[2]*counts_neutrino[i,j])**2/counts_RD[i,j]
            elif np.max([counts_noise[i,j],counts_muon[i,j],counts_neutrino[i,j]]) >0:
                sum+= (counts_RD[i,j] - N_scalers[0]*counts_noise[i,j] - N_scalers[1]*counts_muon[i,j] - N_scalers[2]*counts_neutrino[i,j])**2/np.max([counts_noise[i,j],counts_muon[i,j],counts_neutrino[i,j]])
            else:
                sum+=0
    return sum

N_0 = [50000,50000,50000]
res = optimize.minimize(chi_square,N_0,bounds = ((0, None),(0, None), (0, None)))
print(res.x)
print(res.success)

N_test = [500000,500000,50000]


counts_noise_fit = counts_noise*res.x[0]#*N_test[0]
counts_muon_fit = counts_muon*res.x[1]#*N_test[1]#
counts_neutrino_fit = counts_neutrino*res.x[2]#*N_test[2]#

#print('noise:',counts_noise_fit[:10])
#print('muon:',counts_muon_fit[:10])
#print('neutrino:',counts_neutrino_fit[:10])

counts_residual = counts_noise_fit + counts_muon_fit + counts_neutrino_fit - counts_RD

fig, axs = plt.subplots(figsize=(8, 8))
im = axs.imshow(counts_residual,norm=colors.LogNorm())
fig.colorbar(im, ax=axs)
fig.tight_layout()

fig.savefig(outdir + 'residual_heatmap')

fig, axs = plt.subplots(figsize=(20, 10))

sum_axis=0
bin_width = bins_to_use[1]-bins_to_use[0]

axs.bar(bins_middle,np.sum(counts_noise_fit,axis=sum_axis),width=bin_width,label='Scaled Noise')
axs.bar(bins_middle,np.sum(counts_muon_fit,axis=sum_axis),width=bin_width,bottom=np.sum(counts_noise_fit,axis=sum_axis),label='Scaled Muons')
axs.bar(bins_middle,np.sum(counts_neutrino_fit,axis=sum_axis),width=bin_width,bottom=np.sum(counts_noise_fit+counts_muon_fit,axis=sum_axis),label='Scaled Neutrinos')

#axs.stairs(np.sum(counts_noise_fit,axis=sum_axis), bins_to_use,label='Scaled Noise',fill=True,color='blue')
#axs.stairs(np.sum(counts_muon_fit,axis=sum_axis), bins_to_use,baseline=np.sum(counts_noise_fit,axis=sum_axis),label='Scaled Muons',fill=True,color='orange')
#axs.stairs(np.sum(counts_neutrino_fit,axis=sum_axis), bins_to_use,baseline=np.sum(counts_noise_fit+counts_muon_fit,axis=sum_axis),label='Scaled Neutrinos',fill=True,color='green')


axs.plot(bins_middle,np.sum(counts_RD,axis=sum_axis),'o',label='Real data')
axs.set_xlabel('Neutrino probability')

axs.set_ylabel('Count')
axs.set_yscale('log')
axs.legend(loc='upper right')





fig.tight_layout()

fig.savefig(outdir + 'Scaled_neutrino_prob_histograms.png')


fig, axs = plt.subplots(figsize=(8, 8))

sum_axis=1
bin_width = bins_to_use[1]-bins_to_use[0]

axs.bar(bins_middle,np.sum(counts_noise_fit,axis=sum_axis),width=bin_width,label='Scaled Noise')
axs.bar(bins_middle,np.sum(counts_muon_fit,axis=sum_axis),width=bin_width,bottom=np.sum(counts_noise_fit,axis=sum_axis),label='Scaled Muons')
axs.bar(bins_middle,np.sum(counts_neutrino_fit,axis=sum_axis),width=bin_width,bottom=np.sum(counts_noise_fit+counts_muon_fit,axis=sum_axis),label='Scaled Neutrinos')


#axs.stairs(np.sum(counts_neutrino_fit,axis=sum_axis), bins_to_use,baseline=np.sum(counts_noise_fit+counts_muon_fit,axis=sum_axis),label='Scaled Neutrinos',fill=True,color='green')
#axs.stairs(np.sum(counts_muon_fit,axis=sum_axis), bins_to_use,baseline=np.sum(counts_noise_fit,axis=sum_axis),label='Scaled Muons',fill=True,color='orange')
#axs.stairs(np.sum(counts_noise_fit,axis=sum_axis), bins_to_use,label='Scaled Noise',fill=True,color='blue')


axs.plot(bins_middle,np.sum(counts_RD,axis=sum_axis),'o',label='Real data')
axs.set_xlabel('Noise probability')

axs.set_ylabel('Count')
axs.set_yscale('log')
axs.legend(loc='upper right')

fig.tight_layout()

fig.savefig(outdir + 'Scaled_noise_prob_histograms.png')

print(f'there are predicted noise {np.sum(counts_noise_fit)}' +f' and predicted muons {np.sum(counts_muon_fit)}' +f' and predicted neutrinos {np.sum(counts_neutrino_fit)}' )

