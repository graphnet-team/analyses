import sqlite3 as sql
#from ..helper_functions.plot_params import *
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_sql
import numpy as np

bin_number = 100
outdir = "/groups/icecube/peter/workspace/analyses/moon_pointing_analysis/plotting/distributions/test_plot/Leon_mu_nu_model_"
azimuth_db = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/trained_models/last_one_lvl3MC/dynedge_azimuth_Leon_mu_neutrino_100000_samples_SRTInIcePulses/results.csv"
azimuth_db = pd.read_csv(azimuth_db)
azimuth = azimuth_db.azimuth_pred
azimuth_real = azimuth_db.azimuth
azimuth_std = 1 / np.sqrt(azimuth_db.azimuth_kappa_pred)

zenith_db = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/trained_models/last_one_lvl3MC/dynedge_zenith_Leon_mu_neutrino_100000_samples_SRTInIcePulses/results.csv"
zenith_db = pd.read_csv(zenith_db)
zenith = zenith_db.zenith_pred
zenith_real = zenith_db.zenith
zenith_std = 1 / np.sqrt(zenith_db.zenith_kappa_pred)
# zenith[zenith>np.pi/2] = np.pi-zenith[zenith>np.pi/2]

to_angles = True
if to_angles == True:
    zenith = zenith * 180 / np.pi
    azimuth = azimuth * 180 / np.pi
    zenith_real = zenith_real * 180 / np.pi
    azimuth_real = azimuth_real * 180 / np.pi

fig, axs = plt.subplots(2, 1, figsize=(16, 8))

axs[0].hist2d(zenith,zenith_real, bin_number)
axs[0].set_title("zenith")
axs[0].set_xlabel("zenith_pred")
axs[0].set_ylabel("zenith_real")

axs[1].hist2d(azimuth,azimuth_real, bin_number)
axs[1].set_title("azimuth")
axs[1].set_xlabel("azimuth_pred")
axs[1].set_ylabel("azimuth_real")


fig.tight_layout()
fig.savefig(
    outdir + "Angular_reconstruction_test_data.png"
)
