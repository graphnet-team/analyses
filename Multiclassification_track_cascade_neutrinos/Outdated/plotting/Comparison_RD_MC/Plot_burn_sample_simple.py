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

indir_RD = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Inference/pid_40_days_sample_first.csv"
outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/burn_sample_"

results_RD = pd.read_csv(indir_RD).sort_values('event_no').reset_index(drop = True)

bins_to_use = np.linspace(0,1,101)
bins_middle = (bins_to_use[1:]+bins_to_use[:-1])/2

print(bins_to_use)
print(bins_middle)


fig, axs = plt.subplots(1,1,sharex=False,figsize=(20, 10))

counts_RD, _,_ = axs.hist(results_RD['pid_neutrino_pred'].values,bins=bins_to_use,label='RD')
print(np.sum(counts_RD))
axs.set_ylabel('Count')

axs.set_yscale('log')

axs.set_xlabel('Neutrino probability')

axs.set_title('Burn Sample RD + MC')

fig.tight_layout()

fig.savefig(outdir + 'Neutrino_prob_histogram.png')




fig, axs = plt.subplots(1,1,sharex=False,figsize=(20, 10))

counts_RD, _,_ = axs.hist(results_RD['pid_muon_pred'].values,bins=bins_to_use,label='RD')

axs.set_ylabel('Count')

axs.set_yscale('log')

axs.set_xlabel('Muon probability')

axs.set_title('Burn Sample RD + MC')

fig.tight_layout()

fig.savefig(outdir + 'Muon_prob_histogram.png')


fig, axs = plt.subplots(1,1,sharex=False,figsize=(20, 10))

counts_RD, _,_ = axs.hist(results_RD['pid_noise_pred'].values,bins=bins_to_use,label='RD')

axs.set_ylabel('Count')

axs.set_yscale('log')

axs.set_xlabel('Noise probability')

axs.set_title('Burn Sample RD + MC')

fig.tight_layout()

fig.savefig(outdir + 'Noise_prob_histogram.png')