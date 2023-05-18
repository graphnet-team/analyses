import sqlite3 as sql

import numpy as np
import pandas as pd
from pandas import cut, read_sql
import pickle as pkl
from random import choices
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors as colors

indir = "/groups/icecube/peter/storage/Multiclassification/Real/last_one_lvl3MC/dynedge_pid_Real_run_21.5_mill_equal_frac_second_/results.csv"
outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/_25.5M"

results = pd.read_csv(indir).sort_values('event_no').reset_index(drop = True)

#print(results.head(10))
#Noise = 0, muon =1, Neutrino=2
pid_transform = {1:0,12:2,13:1,14:2,16:2}

predictions = []
truth = []

noise = 0
muons = 0
neutrinos = 0 

number = len(results)

for i in range(number):# range(len(results)):
    noise_pred = results['pid_noise_pred'].values[i]
    muon_pred = results['pid_muon_pred'].values[i]
    neutrino_pred = results['pid_neutrino_pred'].values[i]
    predictions.append(np.argmax([noise_pred,muon_pred,neutrino_pred]))

    truth.append(pid_transform[abs(results['pid'].values[i])])
    if pid_transform[abs(results['pid'].values[i])] == 0:
        noise+=1
    if pid_transform[abs(results['pid'].values[i])] == 1:
        muons+=1
    if pid_transform[abs(results['pid'].values[i])] == 2:
        neutrinos+=1

print(f'there are {noise} noise, {muons} muons, and {neutrinos} neutrinos')

confusion_matrix = metrics.confusion_matrix(truth, predictions)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Noise','Muons','Neutrinos'])
cm_display.plot()
plt.show()
plt.savefig(outdir + 'Confusion_matrix.png')

bins = 50

fig, axs = plt.subplots(3,1,sharex=False,figsize=(8, 16))

mask_noise = [True if truth[i] ==0 else False for i in range(len(truth))]
mask_muon = [True if truth[i] ==1 else False for i in range(len(truth))]
mask_neutrino = [True if truth[i] ==2 else False for i in range(len(truth))]

cd1 = axs[0].hist2d(results['pid_noise_pred'].values[:number][mask_noise],results['pid_neutrino_pred'].values[:number][mask_noise],bins,norm=colors.LogNorm())
cd2 = axs[1].hist2d(results['pid_noise_pred'].values[:number][mask_muon],results['pid_neutrino_pred'].values[:number][mask_muon],bins,norm=colors.LogNorm())
cd3 = axs[2].hist2d(results['pid_noise_pred'].values[:number][mask_neutrino],results['pid_neutrino_pred'].values[:number][mask_neutrino],bins,norm=colors.LogNorm())


axs[0].set_ylabel('Neutrino probability')
axs[1].set_ylabel('Neutrino probability')
axs[2].set_ylabel('Neutrino probability')

axs[0].set_xlabel('Noise probability')
axs[1].set_xlabel('Noise probability')
axs[2].set_xlabel('Noise probability')

axs[0].set_title('Noise')
axs[1].set_title('Muons')
axs[2].set_title('Neutrinos')

fig.colorbar(cd1[3], ax=axs[0])
fig.colorbar(cd2[3], ax=axs[1])
fig.colorbar(cd3[3], ax=axs[2])



fig.tight_layout()

fig.savefig(outdir + 'prob_heatmaps.png')


fpr_neutrino, tpr_neutrino , _ = metrics.roc_curve(truth,results['pid_neutrino_pred'].values,pos_label=2)
fpr_muon, tpr_muon , _ = metrics.roc_curve(truth,results['pid_muon_pred'].values,pos_label=1)
fpr_noise, tpr_noise , _ = metrics.roc_curve(truth,results['pid_noise_pred'].values,pos_label=0)

auc_neutrino = metrics.auc(fpr_neutrino, tpr_neutrino)
auc_muon = metrics.auc(fpr_muon, tpr_muon)
auc_noise = metrics.auc(fpr_noise, tpr_noise)


fig, axs = plt.subplots(figsize=(16, 8))

axs.plot(fpr_neutrino, tpr_neutrino, label=f'Neutrino Auc={auc_neutrino}')
axs.plot(fpr_muon, tpr_muon, label=f'Muon Auc={auc_muon}')
axs.plot(fpr_noise, tpr_noise, label=f'Noise Auc={auc_noise}')

axs.set_xlabel('FPR')
axs.set_ylabel('TPR')
axs.set_title('ROC curves')


fig.tight_layout()

fig.savefig(outdir + 'Roc_curves.png')


