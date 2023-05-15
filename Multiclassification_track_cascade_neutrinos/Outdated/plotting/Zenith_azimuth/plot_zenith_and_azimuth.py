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

outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/Zenith_MP_database/"

indir_test_zenith = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track/osc_next_level3_v2/Peter_Morten_zenith_test_1_mill_test_set/results.csv"
indir_test_azimuth = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track/osc_next_level3_v2/Peter_Morten_azimuth_test_test_set/results.csv"

results_test_zenith = pd.read_csv(indir_test_zenith).sort_values('event_no').reset_index(drop = True)
results_test_azimuth = pd.read_csv(indir_test_azimuth).sort_values('event_no').reset_index(drop = True)

fig, axs = plt.subplots(figsize=(8, 8))

kappa_cut = 20

axs.hist2d(results_test_zenith['zenith_pred'][results_test_zenith['zenith_kappa']>kappa_cut],results_test_zenith['zenith'][results_test_zenith['zenith_kappa']>kappa_cut],bins=100)

axs.set_ylabel('True Zenith')
axs.set_xlabel('Predicted Zenith')

fig.tight_layout()

fig.savefig(outdir + 'Zenith_pred_2d_hist.png')




fig, axs = plt.subplots(figsize=(8, 8))


axs.hist2d(results_test_azimuth['azimuth_pred'][results_test_azimuth['azimuth_kappa']>kappa_cut],results_test_azimuth['azimuth'][results_test_azimuth['azimuth_kappa']>kappa_cut],bins=100)

axs.set_ylabel('True azimuth')
axs.set_xlabel('Predicted azimuth')

fig.tight_layout()

fig.savefig(outdir + 'Azimuth_pred_2d_hist.png')