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

indir = "/groups/icecube/petersen/GraphNetDatabaseRepository/dev_lvl3_genie_burnsample/dev_lvl3_genie_burnsample_v5.db"

#Load in truth data
with sql.connect(indir) as con:
    query = """
    SELECT
        event_no, zenith, sim_type
    FROM 
        truth
    """
    truth_data = read_sql(query,con)

print(truth_data.head(10))

print(truth_data['sim_type'].unique())

RD_event_no = truth_data['event_no'][(truth_data['sim_type'] == "data")]
MC_event_no = truth_data['event_no'][(truth_data['sim_type'] != "data")]

RD_event_no.to_csv('dev_lvl3_genie_burnsample_RD_event_numbers.csv',index=False)
MC_event_no.to_csv('dev_lvl3_genie_burnsample_MC_event_numbers.csv',index=False)