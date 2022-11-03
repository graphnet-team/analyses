import sqlite3 as sql
from plot_params import *

import pandas as pd
from pandas import read_sql
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

outdir = '/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/plots/comparison_to_moon/_Leon_mu_nu_1_million_'

real_data = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/real_data/data_with_reco/moonL4_segspline_exp13_01_merged_with_time_and_reco_and_new_pulsemap.db"

prediction_zenith_dir = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/Inference/zenith_Leon_MC_mu_nu_1000000_TWSRTHV_predictions.csv"
prediction_azimuth_dir = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/Inference/azimuth_Leon_MC_mu_nu_1000000_TWSRTHV_predictions.csv"
prediction_zenith_data = pd.read_csv(prediction_zenith_dir).sort_values('event_no')#.reset_index(drop = True)
prediction_azimuth_data = pd.read_csv(prediction_azimuth_dir).sort_values('event_no')#.reset_index(drop = True)
#print(prediction_zenith_data.head(10))
azimuth_prediction = prediction_azimuth_data['azimuth_pred']
zenith_prediction = prediction_zenith_data['zenith_pred']


with sql.connect(real_data) as con:
    query_moon = """
    SELECT
        azimuth, zenith
    FROM 
        MoonDirection;
    """
    moon_position = read_sql(query_moon, con)

    query_event_no = """
    SELECT
        event_no
    FROM
        TWSRTHVInIcePulses
    """
    full_event_no = read_sql(query_event_no, con)

    query_spline_mpe_ic = """
    SELECT
        azimuth_spline_mpe_ic, event_no, zenith_spline_mpe_ic
    FROM 
        spline_mpe_ic;
    """
    spline_mpe_ic = read_sql(query_spline_mpe_ic, con)



moon_position = pd.concat([moon_position,full_event_no],axis=1)
moon_position.drop_duplicates(subset=['event_no'],keep='first',inplace=True)

#spline_mpe_ic = spline_mpe_ic[spline_mpe_ic['event_no'].isin(moon_position['event_no'])]
moon_position = moon_position[moon_position['event_no'].isin(spline_mpe_ic['event_no'])]

spline_mpe_ic.sort_values('event_no')
moon_position.sort_values('event_no')

azimuth_moon = moon_position['azimuth'].reset_index(drop=True)
zenith_moon = moon_position['zenith'].reset_index(drop=True)

azimuth_spline_mpe_ic = spline_mpe_ic['azimuth_spline_mpe_ic'].reset_index(drop=True)
zenith_spline_mpe_ic = spline_mpe_ic['zenith_spline_mpe_ic'].reset_index(drop=True)


to_angles = True
if to_angles:
    zenith_moon = zenith_moon * 180 / np.pi
    azimuth_moon = azimuth_moon * 180 / np.pi
    zenith_spline_mpe_ic = zenith_spline_mpe_ic * 180 / np.pi
    azimuth_spline_mpe_ic = azimuth_spline_mpe_ic * 180 / np.pi
    azimuth_prediction = azimuth_prediction *180 / np.pi
    zenith_prediction = zenith_prediction *180 / np.pi

#good_selection = np.array(zenith_prediction > 0.01) * np.array(zenith_prediction < 170)
good_selection = []
#print(len(good_selection))
print(zenith_prediction)
for i in range(len(zenith_prediction)):
    if np.array(zenith_prediction)[i] > 0.01 and np.array(zenith_prediction)[i] < 170:
        good_selection.append(True)
    else:
        good_selection.append(False)
print(good_selection[:10])


cleaned = False
if cleaned:
    zenith_moon = zenith_moon[good_selection]
    azimuth_moon = azimuth_moon[good_selection]
    zenith_spline_mpe_ic = zenith_spline_mpe_ic[good_selection]
    azimuth_spline_mpe_ic = azimuth_spline_mpe_ic[good_selection]
    azimuth_prediction = azimuth_prediction[good_selection]
    zenith_prediction = zenith_prediction[good_selection]

print(len(azimuth_moon))
print(len(azimuth_spline_mpe_ic))

bin_number=100

fig, axs = plt.subplots(figsize=(16, 8))

hist = axs.hist2d(azimuth_moon, zenith_moon, bin_number,cmap="viridis")
axs.set_title("Moon position")
axs.set_ylabel('zenith')
axs.set_xlabel("azimuth")
fig.colorbar(hist[3], ax=axs)
fig.tight_layout()
fig.savefig(
    outdir + "Moon_position.png"
)

fig, axs = plt.subplots(figsize=(16, 8))

hist = axs.hist2d(azimuth_spline_mpe_ic, zenith_spline_mpe_ic, bin_number,cmap="viridis")
axs.set_title("Spline_mpe_ic reconstructions")
axs.set_ylabel('zenith')
axs.set_xlabel("azimuth")
fig.colorbar(hist[3], ax=axs)
fig.tight_layout()
fig.savefig(
    outdir + "Spline_mpe_ic_reconstructions.png"
)

fig, axs = plt.subplots(figsize=(16, 8))

hist = axs.hist2d(azimuth_prediction, zenith_prediction, bin_number,cmap="viridis")
axs.set_title("Dynedge Leon reconstructions, zenith cut at 90")
axs.set_ylabel('zenith')
axs.set_xlabel("azimuth")
fig.colorbar(hist[3], ax=axs)
fig.tight_layout()
fig.savefig(
    outdir + "Dynedge_Leon_reconstructions.png"
)


fig, axs = plt.subplots(figsize=(16, 8))

hist = axs.hist2d(azimuth_prediction, azimuth_spline_mpe_ic, bin_number,cmap="viridis")
axs.set_title("Azimuth")
axs.set_ylabel('Spline MPE')
axs.set_xlabel("Dynedge")
fig.colorbar(hist[3], ax=axs)
fig.tight_layout()
fig.savefig(
    outdir + "Azimuth_dynedge_vs_Spline_MPE_reconstructions.png"
)

fig, axs = plt.subplots(figsize=(16, 8))

hist = axs.hist2d(zenith_prediction, zenith_spline_mpe_ic, bin_number,cmap="viridis")
axs.set_title("zenith")
axs.set_ylabel('Spline MPE')
axs.set_xlabel("Dynedge")
fig.colorbar(hist[3], ax=axs)
fig.tight_layout()
fig.savefig(
    outdir + "Zenith_dynedge_vs_Spline_MPE_reconstructions.png"
)



azimuth_delta_spline_moon = azimuth_spline_mpe_ic - azimuth_moon
zenith_delta_spline_moon = zenith_spline_mpe_ic - zenith_moon

if to_angles:
    azimuth_delta_spline_moon[azimuth_delta_spline_moon > 180] = -360 + azimuth_delta_spline_moon[azimuth_delta_spline_moon > 180]
    azimuth_delta_spline_moon[azimuth_delta_spline_moon < -180] = 360 + azimuth_delta_spline_moon[azimuth_delta_spline_moon < -180]
else:
    azimuth_delta_spline_moon[azimuth_delta_spline_moon > np.pi] =-2*np.pi + azimuth_delta_spline_moon[azimuth_delta_spline_moon > np.pi]
    azimuth_delta_spline_moon[azimuth_delta_spline_moon < -np.pi] = 2*np.pi + azimuth_delta_spline_moon[azimuth_delta_spline_moon < -np.pi]


fig, axs = plt.subplots(figsize=(16, 8))

hist = axs.hist2d(azimuth_delta_spline_moon, zenith_delta_spline_moon, bin_number,cmap="viridis")
axs.set_title("reconstructions minus moon positions")
axs.set_ylabel('delta zenith')
axs.set_xlabel("delta azimuth")
fig.colorbar(hist[3], ax=axs)
fig.tight_layout()
fig.savefig(
    outdir + "delta_Spline_mpe_ic_moon.png"
)

#bad_selection_mask = []
#for i in range(len(zenith_prediction)):
#    if zenith_prediction[i]<0.01 or zenith_prediction[i]>170:
#        bad_selection_mask.append(i)

#print(bad_selection_mask[:10])

