"""mostly an example script"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import sqlite3 as sql
import argparse
from pandas import read_sql
from helper_functions.plot_params import *
import numpy as np

parser = argparse.ArgumentParser(
    description="processing i3 files to sqlite3 databases"
)
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_db",
    type=str,
    help="path to database [str]",
    default = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/Leon_MC_data/last_one_lvl3MC.db",
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="the output path [str]",
    default = "/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/plots/distributions/",
)

args = parser.parse_args()

tag = 'Leon_MC_data_'

# dataloading
with sql.connect(args.path_to_db) as con:
    query = """
    SELECT
        azimuth, zenith, energy, pid, event_no
    FROM 
        truth 
    LIMIT 
        10000000;
    """ 
    sql_data = read_sql(query, con)

print(sql_data['pid'].unique())
pids = [13,14,12,16]

plt.figure()
for i in pids:
    print(i)
    if i==-1:
        break
    plt.hist(np.log10(np.array(sql_data[np.abs(sql_data['pid'])==i]["energy"])), bins=100,alpha=0.5,label=f'{i}')
plt.yscale("log")
plt.title("input data: Energy in log10")
plt.legend()
plt.savefig(args.output + tag + 'energy' ".png")

plt.figure()
for i in pids:
    plt.hist(sql_data[np.abs(sql_data['pid'])==i]["azimuth"]*180/np.pi, bins=100,alpha=0.5,label=f'{i}')
plt.yscale("log")
plt.title("input data: azimuth")
plt.legend()
plt.savefig(args.output + tag + 'azimuth' ".png")

plt.figure()
for i in pids:
    plt.hist(sql_data["zenith"][np.abs(sql_data['pid'])==i]*180/np.pi, bins=100,alpha=0.5,label=f'{i}')
plt.yscale("log")
plt.title("input data: zenith")
plt.legend()
plt.savefig(args.output + tag + 'zenith' ".png")