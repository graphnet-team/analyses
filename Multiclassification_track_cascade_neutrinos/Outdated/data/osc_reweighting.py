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
import torch


db_path = "/groups/icecube/petersen/GraphNetDatabaseRepository/Leon2022_DataAndMC_CSVandDB_StoppedMuons/last_one_lvl3MC.db"

with sql.connect(db_path) as con:
    query = f"""
    SELECT
        event_no, osc_weight
    FROM 
        retro;
    """
    weights = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)

with sql.connect(db_path) as con:
    query = f"""
    SELECT
        event_no, pid
    FROM 
        truth;
    """
    truth = pd.read_sql(query,con).sort_values('event_no').reset_index(drop = True)

truth_and_weights = pd.concat([truth, weights['osc_weight']], axis=1)

def get_weights_Peter_Morten(df, n_files):
    ratio_dict = dict(zip([1,13,12,14,16], n_files))
    df['n_files'] = df['pid'].abs().map(ratio_dict)
    df['normalised_weights'] = df['osc_weight'] / df['n_files']
    return df

n_files_last_one_lvl3MC = [10000,2001,186,213,119]
n_files_Peter_og_Morten = [7001,1775,602,1518,334]

truth_and_normalised_weights = get_weights_Peter_Morten(truth_and_weights,n_files_last_one_lvl3MC)
