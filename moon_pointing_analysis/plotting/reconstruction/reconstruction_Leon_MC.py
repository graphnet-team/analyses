from plot_params import *
import pandas as pd
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
    default="/groups/icecube/peter/storage/MoonPointing/Models/Leon_Muon_data_MC/last_one_lvl3MC/dynedge_zenith_Leon_muon_data_MC/results.csv",
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="the output path [str]",
    default="/groups/icecube/qgf305/workspace/analyses/moon_pointing_analysis/plotting/reconstruction/test_plot",
)
parser.add_argument(
    "-b",
    "--bins",
    dest="bins",
    type=int,
    help="the number of bins [str]",
    default=25,
)
args = parser.parse_args()

# azimuth = ""
# azimuth = pd.read_csv(azimuth)
# print(len(azimuth.azimuth_pred))
# plt.figure()
# plt.hist2d(azimuth.azimuth_pred, azimuth.azimuth, bins = bin_number,cmap='viridis')
# plt.title("results: zenith prediction")
# plt.colorbar()
# plt.legend()
# plt.savefig("/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/plotting/Test_Plots/Leon_MC_Muon_data/azimuth_true_vs_pred.png")
#
zenith = "/groups/icecube/peter/storage/MoonPointing/Models/Leon_Muon_data_MC/last_one_lvl3MC/dynedge_zenith_Leon_muon_data_MC/results.csv"
zenith = pd.read_csv(zenith)

plt.figure()
plt.hist2d(zenith.zenith_pred, zenith.zenith, bins=args.bins, cmap="viridis")
plt.title("results: zenith prediction")
plt.xlabel("zenith prediction")
plt.xlim((0, 1.5))
plt.ylabel("true zenith")
plt.colorbar()
plt.legend()
plt.savefig(args.output + "zenith_true_vs_pred.png")
