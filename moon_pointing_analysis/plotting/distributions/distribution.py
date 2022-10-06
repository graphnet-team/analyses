import sqlite3 as sql
#from plot_params import *

from plot_params import *

import pandas as pd
from pandas import read_sql
import numpy as np

# data pathing
indir = "/groups/icecube/qgf305/storage/test/Saskia_datapipeline/L2_2018_1/Sebastian_MoonDataL4/dynedge_zenith_predict_zenith/"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/distributions/test_plot/"

inferenceResults = 'results.csv'
df = pd.read_csv(indir+inferenceResults)

plt.figure(figsize=single)
plt.hist(df.zenith_pred, bins = 10, label="zenith predictions")
plt.title("results: zenith prediction")
plt.yscale('log')
plt.legend()
plt.savefig(outdir+"inferenceResults.png")