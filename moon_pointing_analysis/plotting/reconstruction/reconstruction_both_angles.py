from plot_params import *
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="plotting the predicted zenith and azimuth vs truth."
)
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_csv",
    type=str,
    help="path to database [str]",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="the output path [str]",
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

database = pd.read_csv(args.path_to_csv)

for item in ["zenith", "azimuth"]:
    plt.figure()
    plt.hist2d(
        database[f"{eval(item)}_pred"],
        database[eval(item)],
        bins=args.bins,
        cmap="viridis",
    )
    plt.title(f"results: {item} prediction")
    plt.colorbar()
    plt.legend()
    plt.savefig(args.output + item + "_results.png")

plt.figure()
plt.hist(1 / np.sqrt(database["kappa_pred"]), bins=args.bins)
plt.title("results: kappa prediction")
plt.legend()
plt.savefig(args.output + "kappa_results.png")
