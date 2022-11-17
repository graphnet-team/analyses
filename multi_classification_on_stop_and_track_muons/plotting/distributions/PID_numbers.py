import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import argparse
import sqlite3 as sql
from pandas import read_sql
from helper_functions.plot_params import * 
import numpy as np

indir = "/groups/icecube/leonbozi/work2/data/lvl3_dbs/last_one_lvl3MC/data/last_one_lvl3MC.db"
outdir = '/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/'
# dataloading
with sql.connect(indir) as con:
    query = """
    SELECT
        pid
    FROM 
        truth;
    """ 
    
    sql_data = read_sql(query, con)

print(sql_data.groupby('pid').size())

bins = np.linspace(-20,20,42)
plt.figure()
plt.hist(sql_data['pid'].values,bins)
plt.savefig(outdir + 'pid_hist.png')
