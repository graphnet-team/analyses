import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import pandas as pd
import numpy as np
import argparse
import math
from helper_functions.plot_params import *

parser = argparse.ArgumentParser(
    description="plotting the predicted zenith and azimuth vs truth."
)
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_csv",
    nargs="+",
    help="<required> path(s) to database [list]",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="<required> the output path [str]",
    required=True,
)
parser.add_argument(
    "-b",
    "--bins",
    dest="bins",
    type=int,
    help="the number of bins [str]",
    default=25,
)
parser.add_argument(
    "-p",
    "--pulsemap",
    dest="pulsemap",
    type=str,
    help="the pulsemap used [str]",
    required=True,
)
args = parser.parse_args()

def rad_to_deg(df):
    return 180*df/math.pi

# pattern to look for
patterns = ["zenith", "azimuth"]
# name of the database
db_names = ["zenith_db", "azimuth_db"]
# combination of pattern and database name
for pattern, db_name in zip(patterns, db_names):
    for pathing in args.path_to_csv:
        # check if string contains the pattern
        if pattern in pathing:
            # write a local variable database callable
            locals()[db_name] = pd.read_csv(pathing)


    try:
        plt.figure()
        # make histogram of the pattern in the given db_name
        mappable = plt.hist2d(
            rad_to_deg(eval(db_name)[pattern+"_pred"]),
            rad_to_deg(eval(db_name)[pattern]),
            bins=args.bins,
            cmap="viridis",
        )
        plt.title(f"results: {pattern} prediction")
        colorbar(mappable[3])
        plt.xlabel(pattern+"_pred")
        plt.ylabel(pattern)
        plt.legend()
        plt.savefig(args.output + pattern + "_results.png")
    except:
        print(pattern + " failed")

    try:
        plt.figure()
        plt.hist(1 / np.sqrt(eval(db_name)[pattern+"_kappa_pred"]), bins=args.bins)
        plt.title("results: kappa prediction")
        plt.yscale('log')
        plt.legend()
        plt.savefig(args.output + pattern + "_kappa_results.png")
    except:
        print(pattern + " failed for kappa")
