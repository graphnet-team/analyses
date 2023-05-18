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

indir_test_track_cascade = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models/osc_next_level3_v2/dynedge_track_mu_Track_cascade_osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track_SplitInIcePulses_on_all_neutrinos_test/results.csv"

results_test_track_cascade = pd.read_csv(indir_test_track_cascade).sort_values('event_no').reset_index(drop = True)

bins = np.linspace(0,1,100)

fig, axs = plt.subplots(figsize=(8, 8))


axs.hist(results_test_track_cascade['track_mu_pred'][results_test_track_cascade['track_mu']==1.0],bins=bins,label='tracks',alpha=0.5)
axs.hist(results_test_track_cascade['track_mu_pred'][results_test_track_cascade['track_mu']==0.0],bins=bins,label='cascade',alpha=0.5)

axs.set_ylabel('Count')
axs.set_xlabel('track probability')
axs.legend()
fig.tight_layout()

fig.savefig(outdir + 'track_prob_hist.png')
