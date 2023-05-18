import sqlite3 as sql

import numpy as np
import pandas as pd
from pandas import cut, read_sql
import pickle as pkl
from random import choices
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors as colors

indir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Inference/pid_Leon_RD_results_new_model.csv"
outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/_RD"

results = pd.read_csv(indir).sort_values('event_no').reset_index(drop = True)

#print(results.head(10))
#Noise = 0, muon =1, Neutrino=2
pid_transform = {1:0,12:2,13:1,14:2,16:2}

predictions = []


noise = 0
muons = 0
neutrinos = 0 

number = len(results)

for i in range(number):# range(len(results)):
    noise_pred = results['pid_noise_pred'].values[i]
    muon_pred = results['pid_muon_pred'].values[i]
    neutrino_pred = results['pid_neutrino_pred'].values[i]
    predictions.append(np.argmax([noise_pred,muon_pred,neutrino_pred]))

    
    if predictions[i] == 0:
        noise+=1
    if predictions[i] == 1:
        muons+=1
    if predictions[i] == 2:
        neutrinos+=1

print(f'there are predicted: {noise} noise, {muons} muons, and {neutrinos} neutrinos')


bins = np.linspace(0,50,51)
print(bins)

fig, axs = plt.subplots(figsize=(8, 8))

cd1 = axs.hist2d(results['pid_noise_pred'].values[:number],results['pid_neutrino_pred'].values[:number],bins,norm=colors.LogNorm())

axs.set_ylabel('Neutrino probability')
axs.set_xlabel('Noise probability')
axs.set_title('RD Leon')

fig.colorbar(cd1[3], ax=axs)
fig.tight_layout()
fig.savefig(outdir + 'prob_heatmaps.png')









