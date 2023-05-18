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
from standard_plotting import set_size

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": False,
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 11,
    "font.size": 11,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
}

plt.rcParams.update(tex_fonts)


MC_database_indir = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"
RD_database_indir = "/groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"
lvl3_variables_MC_indir = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db_3/NOT_full_db_osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_truth_and_lvl3_variables.db"
lvl3_variables_RD_indir = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_0.01_percent_burnsample_Peter/merged_db/Burnsample_lvl3_v02.00_lvl3_variables.db"


with sql.connect(RD_database_indir) as con:
        query = f"""
        SELECT
            event_no, RunID, SubRunID, EventID, SubEventID
        FROM 
            truth
        """
        truth_RD = read_sql(query,con)

with sql.connect(lvl3_variables_RD_indir) as con:
        query = f"""
        SELECT
            event_no, RunID, SubRunID, EventID, SubEventID, C2HR6, CausalVetoHits, CleanedFullTimeLength, DCFiducialHits, L3_oscNext_bool, NAbove200Hits, NchCleaned, NoiseEngineNoCharge, RTVetoCutHit, STW9000_DTW300Hits, UncleanedFullTimeLength, VertexGuessZ, VetoFiducialRatioHits
        FROM 
            truth
        """
        lvl3_variables_RD = read_sql(query,con)



print(len(truth_RD))
lvl3_variables_RD = lvl3_variables_RD.drop(columns='event_no')
lvl3_variables_RD_to_save = pd.merge(truth_RD.reset_index(drop=True),lvl3_variables_RD.reset_index(drop=True),on=['RunID', 'SubrunID', 'EventID', 'SubEventID'],how='inner')
print(len(lvl3_variables_RD_to_save))
RD_save_dir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Burnsample/"
lvl3_variables_RD_to_save.to_csv(RD_save_dir + 'lvl3_variables_burnsample.csv')