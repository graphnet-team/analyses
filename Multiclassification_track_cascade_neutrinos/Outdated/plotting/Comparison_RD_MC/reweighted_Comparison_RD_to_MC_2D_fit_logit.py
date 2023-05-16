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
import torch

indir_RD = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Inference/pid_Leon_RD_results_new_model.csv"
indir_MC = "/groups/icecube/peter/storage/Multiclassification/Real/last_one_lvl3MC/dynedge_pid_Real_run_21.5_mill_equal_frac_second_/results.csv"
outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/Reweighed_comparison/reweighted_2d_comparison_logit_"
indir_MC_database = "/groups/icecube/petersen/GraphNetDatabaseRepository/Leon2022_DataAndMC_CSVandDB_StoppedMuons/last_one_lvl3MC.db"

load_reco_weights = True


if not load_reco_weights:
    #input_features = FEATURES.DEEPCORE , rde, pmt_area
    with sql.connect(indir_MC_database) as con:
        query = f"""
        SELECT
            event_no, osc_weight
        FROM 
            retro;
        """
        weights = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)

    weights.to_csv('weight_data.csv')
else:
    weights = pd.read_csv('/groups/icecube/peter/workspace/analyses/multi_classification_on_stop_and_track_muons/plotting/Comparison_RD_MC/weight_data.csv')
    weights = weights.sort_values('event_no').reset_index(drop = True)



results_RD = pd.read_csv(indir_RD).sort_values('event_no').reset_index(drop = True)
results_MC = pd.read_csv(indir_MC).sort_values('event_no').reset_index(drop = True)


weights = weights[weights.event_no.isin(results_MC['event_no'])].sort_values('event_no').reset_index(drop = True)

bins_to_use = np.linspace(-17,17,100)
bins_to_fit = np.linspace(-15,15,100) #bins_to_use
bins_middle = (bins_to_use[1:]+bins_to_use[:-1])/2

#print(bins_to_use)
#print(bins_middle)

pid_transform = {1:0,12:2,13:1,14:2,16:2}


truth_MC = []

for i in range(len(results_MC)):# range(len(results)):
    truth_MC.append(pid_transform[abs(results_MC['pid'].values[i])])



weights_noise = weights[weights.event_no.isin(results_MC['event_no'][np.array(truth_MC) == 0])].sort_values('event_no').reset_index(drop = True)
weights_muon = weights[weights.event_no.isin(results_MC['event_no'][np.array(truth_MC) == 1])].sort_values('event_no').reset_index(drop = True)
weights_neutrino = weights[weights.event_no.isin(results_MC['event_no'][np.array(truth_MC) == 2])].sort_values('event_no').reset_index(drop = True)


class_options = {1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2}
pid_transform_morten = torch.tensor([class_options[int(value)] for value in results_MC["pid"]])

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

def inverse_logit(p):
    eps = 0.0000001
    try:
        if np.isnan(p):
            return
        p = p*(1-2*eps)+eps
        logit = np.log(np.exp(p)/(1+np.exp(p)))
    except ZeroDivisionError:
        print(e)
    return logit



keys = ["noise", "muon", "neutrino"]
noise_logits, muon_logits, neutrino_logits = dict(), dict(), dict()
mask = dict()
# loop over dataset
for key in keys:
    # loop over noise/particle type
    for class_type, item in enumerate(keys):
        mask[item] = results_MC[f'pid_{key}_pred'].where(pid_transform_morten == class_type)
        print(f"{key}_logits", item)
        locals()[f"{key}_logits"][item] = pd.Series(mask[item]).apply(to_logit)
    locals()[f"{key}_logits"]["RD"] = pd.Series(results_RD[f'pid_{key}_pred']).apply(to_logit)

#noise_logits, muon_logits, neutrino_logits = dict(), dict(), dict()
#mask = dict()
#for i, item in enumerate(["noise", "muon", "neutrino"]):
#    mask[item] = results_MC[f'pid_noise_pred'].where(pid_transform_morten == i)
#    noise_logits[item] = pd.Series(mask[item]).apply(to_logit)
#
#mask = dict()
#for i, item in enumerate(["noise", "muon", "neutrino"]):
#    mask[item] = results_MC[f'pid_muon_pred'].where(pid_transform_morten == i)
#    muon_logits[item] = pd.Series(mask[item]).apply(to_logit)
#
#mask = dict()
#for i, item in enumerate(["noise", "muon", "neutrino"]):
#    mask[item] = results_MC[f'pid_neutrino_pred'].where(pid_transform_morten == i)
#    neutrino_logits[item] = pd.Series(mask[item]).apply(to_logit)
#
#noise_logits['RD'] = pd.Series(results_RD[f'pid_noise_pred']).apply(to_logit)
#muon_logits['RD'] = pd.Series(results_RD[f'pid_muon_pred']).apply(to_logit)
#neutrino_logits['RD'] = pd.Series(results_RD[f'pid_neutrino_pred']).apply(to_logit)
fig, axs = plt.subplots(3,1,sharex=False,figsize=(8, 32))
axs[0].hist(neutrino_logits['noise'],bins_to_fit,density=True,label='no scaling',alpha=0.5)
axs[1].hist(neutrino_logits['muon'],bins_to_fit,density=True,label='no scaling',alpha=0.5)
axs[2].hist(neutrino_logits['neutrino'],bins_to_fit,density=True,label='no scaling',alpha=0.5)
axs[0].hist(neutrino_logits['noise'],bins_to_fit,density=True,weights=weights['osc_weight'],label='osc_weight scaling',alpha=0.5)
axs[1].hist(neutrino_logits['muon'],bins_to_fit,density=True,weights=weights['osc_weight'],label='osc_weight scaling',alpha=0.5)
axs[2].hist(neutrino_logits['neutrino'],bins_to_fit,density=True,weights=weights['osc_weight'],label='osc_weight scaling',alpha=0.5)

axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[2].set_yscale('log')

axs[0].legend()
axs[1].legend()
axs[2].legend()

axs[0].set_xlabel('Neutrino probability')
axs[1].set_xlabel('Neutrino probability')
axs[2].set_xlabel('Neutrino probability')

axs[0].set_title('MC Noise')
axs[1].set_title('MC Muons')
axs[2].set_title('MC Neutrino')


fig.tight_layout()

fig.savefig(outdir + '1D_histograms_before_after_weighing.png')



#mask_noise = [True if truth_MC[i] ==0 else False for i in range(len(truth_MC))]
#mask_muon = [True if truth_MC[i] ==1 else False for i in range(len(truth_MC))]
#mask_neutrino = [True if truth_MC[i] ==2 else False for i in range(len(truth_MC))]
fig, axs = plt.subplots(4,1,sharex=False,figsize=(8, 32))
counts_noise_to_fit,_,_,_ = axs[0].hist2d(noise_logits['noise'],neutrino_logits['noise'],bins_to_fit,weights=weights['osc_weight'],norm=colors.LogNorm())
counts_muon_to_fit,_,_,_ = axs[1].hist2d(noise_logits['muon'],neutrino_logits['muon'],bins_to_fit,weights=weights['osc_weight'],norm=colors.LogNorm())
counts_neutrino_to_fit,_,_,_ = axs[2].hist2d(noise_logits['neutrino'],neutrino_logits['neutrino'],bins_to_fit,weights=weights['osc_weight'],norm=colors.LogNorm())
counts_RD_to_fit,_,_,_ = axs[3].hist2d(noise_logits['RD'],neutrino_logits['RD'],bins_to_fit,norm=colors.LogNorm())


fig, axs = plt.subplots(4,1,sharex=False,figsize=(8, 32))

counts_noise,_,_,_ = axs[0].hist2d(noise_logits['noise'],neutrino_logits['noise'],bins_to_use,norm=colors.LogNorm(),weights=weights['osc_weight'])
counts_muon,_,_,_ = axs[1].hist2d(noise_logits['muon'],neutrino_logits['muon'],bins_to_use,norm=colors.LogNorm(),weights=weights['osc_weight'])
counts_neutrino,_,_,_ = axs[2].hist2d(noise_logits['neutrino'],neutrino_logits['neutrino'],bins_to_use,norm=colors.LogNorm(),weights=weights['osc_weight'])
counts_RD,_,_,_ = axs[3].hist2d(noise_logits['RD'],neutrino_logits['RD'],bins_to_use,norm=colors.LogNorm())

axs[0].set_xlabel('noise probability')
axs[1].set_xlabel('noise probability')
axs[2].set_xlabel('noise probability')
axs[3].set_xlabel('noise probability')

axs[0].set_ylabel('Neutrino probability')
axs[1].set_ylabel('Neutrino probability')
axs[2].set_ylabel('Neutrino probability')
axs[3].set_ylabel('Neutrino probability')

axs[0].set_title('MC Noise')
axs[1].set_title('MC Muons')
axs[2].set_title('MC Neutrino')
axs[3].set_title('RD')

fig.tight_layout()

fig.savefig(outdir + '2D_prob_histograms.png')

def chi_square(N_scalers):
    sum = 0
    for i in range(len(bins_to_fit)-1):
        for j in range(len(bins_to_fit)-1):
            if counts_RD_to_fit[i,j] > 0:
                sum+= (counts_RD_to_fit[i,j] - N_scalers[0]*counts_noise_to_fit[i,j] - N_scalers[1]*counts_muon_to_fit[i,j] - N_scalers[2]*counts_neutrino_to_fit[i,j])**2/counts_RD_to_fit[i,j]
            elif np.max([counts_noise_to_fit[i,j],counts_muon_to_fit[i,j],counts_neutrino_to_fit[i,j]]) >0:
                sum+= (counts_RD_to_fit[i,j] - N_scalers[0]*counts_noise_to_fit[i,j] - N_scalers[1]*counts_muon_to_fit[i,j] - N_scalers[2]*counts_neutrino_to_fit[i,j])**2/np.sum([counts_noise_to_fit[i,j],counts_muon_to_fit[i,j],counts_neutrino_to_fit[i,j]])
            else:
                sum+=0
    return sum

N_0 = [100,100,10000000]
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

counts_residual = (counts_noise_fit + counts_muon_fit + counts_neutrino_fit - counts_RD)**2
print('total residual = ')
print(np.sum(counts_residual))

fig, axs = plt.subplots(figsize=(8, 8))
im = axs.imshow(counts_residual,norm=colors.LogNorm())
fig.colorbar(im, ax=axs)
fig.tight_layout()

fig.savefig(outdir + 'residual_heatmap')

fig, axs = plt.subplots(figsize=(8, 8))

sum_axis=0
bin_width = bins_to_use[1]-bins_to_use[0]

axs.bar(bins_middle,np.sum(counts_noise_fit,axis=sum_axis),width=bin_width,label='Scaled Noise')
axs.bar(bins_middle,np.sum(counts_muon_fit,axis=sum_axis),width=bin_width,bottom=np.sum(counts_noise_fit,axis=sum_axis),label='Scaled Muons')
axs.bar(bins_middle,np.sum(counts_neutrino_fit,axis=sum_axis),width=bin_width,bottom=np.sum(counts_noise_fit+counts_muon_fit,axis=sum_axis),label='Scaled Neutrinos')

#axs.stairs(np.sum(counts_noise_fit,axis=sum_axis), bins_to_use,label='Scaled Noise',fill=True,color='blue')
#axs.stairs(np.sum(counts_muon_fit,axis=sum_axis), bins_to_use,baseline=np.sum(counts_noise_fit,axis=sum_axis),label='Scaled Muons',fill=True,color='orange')
#axs.stairs(np.sum(counts_neutrino_fit,axis=sum_axis), bins_to_use,baseline=np.sum(counts_noise_fit+counts_muon_fit,axis=sum_axis),label='Scaled Neutrinos',fill=True,color='green')


axs.plot(bins_middle,np.sum(counts_RD,axis=sum_axis),'o',label='Real data',color='red')
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


axs.plot(bins_middle,np.sum(counts_RD,axis=sum_axis),'o',label='Real data',color='red')
axs.set_xlabel('Noise probability')

axs.set_ylabel('Count')
axs.set_yscale('log')
axs.legend(loc='upper right')

fig.tight_layout()

fig.savefig(outdir + 'Scaled_noise_prob_histograms.png')

print(f'there are predicted noise {np.sum(counts_noise_fit)}' +f' and predicted muons {np.sum(counts_muon_fit)}' +f' and predicted neutrinos {np.sum(counts_neutrino_fit)}' )

