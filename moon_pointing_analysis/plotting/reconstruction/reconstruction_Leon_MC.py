import sqlite3 as sql
from plot_params import *

import pandas as pd
from pandas import read_sql
import numpy as np

bin_number = 50

#azimuth = ""
#azimuth = pd.read_csv(azimuth)
#print(len(azimuth.azimuth_pred))
#plt.figure()
#plt.hist2d(azimuth.azimuth_pred, azimuth.azimuth, bins = bin_number,cmap='viridis')
#plt.title("results: zenith prediction")
#plt.colorbar()
#plt.legend()
#plt.savefig("/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/plotting/Test_Plots/Leon_MC_Muon_data/azimuth_true_vs_pred.png")
#
zenith = "/groups/icecube/peter/storage/MoonPointing/Models/Leon_Muon_data_MC/last_one_lvl3MC/dynedge_zenith_Leon_muon_data_MC/results.csv"
zenith = pd.read_csv(zenith)

plt.figure()
plt.hist2d(zenith.zenith_pred, zenith.zenith, bins = bin_number,cmap='viridis')
plt.title("results: zenith prediction")
plt.xlabel("zenith prediction")
plt.xlim((0,1.5))
plt.ylabel("true zenith")
plt.colorbar()
plt.legend()
plt.savefig("/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/plotting/Test_Plots/Leon_MC_Muon_data/zenith_true_vs_pred.png")


#
#
#inferenceResults = '/groups/icecube/qgf305/storage/test/Saskia_datapipeline/L2_2018_1/Sebastian_MoonDataL4/dynedge_zenith_predict_zenith/results.csv'
#df = pd.read_csv(inferenceResults)
#
#plt.figure()
#plt.hist(df.zenith_pred, bins = 10)
#plt.title("results: zenith prediction")
#plt.yscale('log')
#plt.legend()
#plt.savefig("/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/inferenceResults.png")
#
#db = "/groups/icecube/peter/storage/Sebastian_MoonDataL4.db"
#with sql.connect(db) as con:
#    query = """
#    SELECT
#        charge, dom_time, dom_x, dom_y, dom_z, event_no, pmt_area, rde, width
#    FROM 
#        SRTInIcePulses;
#    """
#    sql_data = read_sql(query,con)
#
#plt.figure()
#plt.hist(sql_data["charge"], bins = 10)
#plt.yscale('log')
#plt.title("input data: Charge")
#plt.legend()
#plt.savefig("/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/L2_2018_1.png")
#
#event_numbers = sql_data["event_no"].unique()
#specific_event = sql_data[sql_data["event_no"] == event_numbers[0]]
#
#fig = plt.figure()
#ax = plt.axes(projection ="3d")
# 
## Creating plot
#dom = ax.scatter3D(
#    specific_event["dom_x"], specific_event["dom_y"], specific_event["dom_z"],
#    c = specific_event["dom_time"], cmap = 'coolwarm', s = 20)
#fig.colorbar(dom, ax=ax)
#ax.set_xlabel("x position")
#ax.set_ylabel("x position")
#ax.set_zlabel('Z Label')
#plt.title(f"simple 3D scatter plot of dom positions for event #{event_numbers[0]}")
#plt.savefig("/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/scatter.png")