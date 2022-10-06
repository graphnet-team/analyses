import sqlite3 as sql
from plot_params import *

import pandas as pd
from pandas import read_sql
import numpy as np

bin_number = 100

azimuth_db = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/azimuth_results.csv"
azimuth_db = pd.read_csv(azimuth_db)
azimuth = azimuth_db.azimuth_pred
azimuth_std = 1/np.sqrt(azimuth_db.azimuth_kappa_pred)

zenith_db = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/zenith_Leon_MC_results.csv"
zenith_db = pd.read_csv(zenith_db)
zenith = zenith_db.zenith_pred
zenith_std = 1/np.sqrt(zenith_db.zenith_kappa_pred)
#zenith[zenith>np.pi/2] = np.pi-zenith[zenith>np.pi/2] 

good_selection_mask = np.array(zenith> 0.001)*np.array(zenith_std<1)*np.array(azimuth_std<1)#np.array(zenith> 0.1)*np.array(zenith_std<1)*np.array(azimuth_std<1)
bad_selection_mask = np.logical_not(good_selection_mask)

plot_first = len(zenith)

to_angles = True
if to_angles == True:
    zenith = zenith*180/np.pi
    azimuth = azimuth*180/np.pi

fig, axs = plt.subplots(2,2,figsize=(16, 8))

axs[0,0].hist(zenith[good_selection_mask][:plot_first],bin_number)
axs[0,0].set_title('good selection')
axs[0,0].set_xlabel("zenith")

axs[1,0].hist(azimuth[good_selection_mask][:plot_first],bin_number)
axs[1,0].set_title('good selection')
axs[1,0].set_xlabel("azimuth")

axs[0,1].hist(zenith[bad_selection_mask][:plot_first],bin_number)
axs[0,1].set_title('bad selection')
axs[0,1].set_xlabel("zenith")

axs[1,1].hist(azimuth[bad_selection_mask][:plot_first],bin_number)
axs[1,1].set_title('bad selection')
axs[1,1].set_xlabel("azimuth")
fig.tight_layout()
fig.savefig("/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/plotting/Test_Plots/Sschindler_L4_data_Leon_model_plots/Angular_reconstruction_test_angles_binned.png")

plt.figure()
plt.hist2d(azimuth[good_selection_mask][:plot_first], zenith[good_selection_mask][:plot_first], bins = bin_number,cmap='viridis')
plt.title("results: angular reconstruction")
plt.xlabel('azimuth')
plt.ylabel('zenith')
plt.colorbar()
plt.legend()
plt.tight_layout()
plt.savefig("/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/plotting/Test_Plots/Sschindler_L4_data_Leon_model_plots/Angular_reconstruction_test.png")


fig, axs = plt.subplots(2,2,figsize=(16, 8))

axs[0,0].hist2d(zenith[good_selection_mask][:plot_first], zenith_std[good_selection_mask][:plot_first], bins = bin_number,cmap='viridis')
axs[0,0].set_ylabel('zenith_std')
axs[0,0].set_xlabel("zenith")


axs[1,0].hist2d(zenith[good_selection_mask][:plot_first], azimuth_std[good_selection_mask][:plot_first], bins = bin_number,cmap='viridis')
axs[1,0].set_ylabel('azimuth_std')
axs[1,0].set_xlabel("zenith")


axs[0,1].hist2d(azimuth[good_selection_mask][:plot_first], azimuth_std[good_selection_mask][:plot_first], bins = bin_number,cmap='viridis')
axs[0,1].set_ylabel('azimuth_std')
axs[0,1].set_xlabel("azimuth")


axs[1,1].hist2d(azimuth[good_selection_mask][:plot_first], zenith_std[good_selection_mask][:plot_first], bins = bin_number,cmap='viridis')
axs[1,1].set_ylabel('zenith_std')
axs[1,1].set_xlabel("azimuth")

fig.tight_layout()
fig.savefig("/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/plotting/Test_Plots/Sschindler_L4_data_Leon_model_plots/Angular_reconstruction_angles_vs_uncertainties.png")


fig, axs = plt.subplots(2,2,figsize=(16, 8))

axs[0,0].hist(zenith_std[good_selection_mask][:plot_first],bin_number)
axs[0,0].set_title('good selection')
axs[0,0].set_xlabel("zenith std")
axs[0,0].set_xlim((0,1))

axs[1,0].hist(azimuth_std[good_selection_mask][:plot_first],bin_number)
axs[1,0].set_title('good selection')
axs[1,0].set_xlabel("azimuth std")
axs[1,0].set_xlim((0,1))

axs[0,1].hist(zenith_std[bad_selection_mask][:plot_first],bin_number)
axs[0,1].set_title('bad selection')
axs[0,1].set_xlabel("zenith std")
axs[0,1].set_xlim((0,1))

axs[1,1].hist(azimuth_std[bad_selection_mask][:plot_first],bin_number)
axs[1,1].set_title('bad selection')
axs[1,1].set_xlabel("azimuth std")
axs[1,1].set_xlim((0,1))
fig.tight_layout()
fig.savefig("/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/plotting/Test_Plots/Sschindler_L4_data_Leon_model_plots/Angular_reconstruction_test_uncertainties_binned.png")





#test = np.array(zenith< 0.1)
#test2 = []
#for i in range(len(test)):
#    if test[i] == True:
#        test2.append(i)
#print(test2)
#print(len(test2))