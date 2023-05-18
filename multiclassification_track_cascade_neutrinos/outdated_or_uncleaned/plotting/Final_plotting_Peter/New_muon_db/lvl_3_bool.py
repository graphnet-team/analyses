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

outdir = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Test_set/"
indir_db_MC = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/Merged_db/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_139008_888003_retro.db"

with sql.connect(indir_db_MC) as con:
        query = f"""
        SELECT
            event_no, L3_oscNext_bool
        FROM 
            truth
        """
        MC_bool = read_sql(query,con)


MC_bool.to_csv(outdir + 'L3_oscNext_bool_new_muon_db.csv',index=False)


outdir_RD = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/Burnsample/"
indir_db_RD = "/groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"

with sql.connect(indir_db_RD) as con:
        query = f"""
        SELECT
            event_no, L3_oscNext_bool
        FROM 
            truth
        """
        RD_bool = read_sql(query,con)


RD_bool.to_csv(outdir_RD + 'Pulses_doms_strings_times_burnsample.csv',index=False)

print('RD saved')