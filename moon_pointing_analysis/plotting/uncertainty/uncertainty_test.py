import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import pandas as pd
import numpy as np
import argparse
from scipy import stats
from scipy.stats import binom, poisson, norm
from helper_functions.plot_params import *
from iminuit import Minuit

from helper_functions.external_functions import Chi2Regression
from helper_functions.external_functions import (
    nice_string_output,
    add_text_to_ax,
    double_gaussian,
    gaussian,
)  # Useful functions to print fit results on figure


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

a = ["/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/trained_models/dynedge_azimuth_example/results.csv",
"/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/trained_models/dynedge_zenith_example/results.csv"]

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

    locals()[db_name]["standard_deviation"] = 1 / np.sqrt(eval(db_name).kappa)
    locals()[db_name]["difference"] = eval(db_name)[pattern+"_pred"] - eval(db_name)[pattern]

    if pattern == "azimuth":
        condition1 = (eval(db_name)["difference"] > np.pi)
        condition2 = (eval(db_name)["difference"] < -np.pi)
        locals()[db_name]["difference"][condition1] = (eval(db_name)["difference"][condition1] - 2*np.pi)
        locals()[db_name]["difference"][condition2] = (eval(db_name)["difference"][condition2] + 2*np.pi)
        
    locals()[db_name][pattern+"_z"] = eval(db_name)["difference"] / eval(db_name)["standard_deviation"]

    counts, bin_edges = np.histogram(eval(db_name)[pattern+"_z"], bins=args.bins)
    
    condition = (counts > 0)
    x = (bin_edges[1:][condition] + bin_edges[:-1][condition]) / 2
    y = counts[condition]
    # NOTE: We (naturally) assume that the bin count is Poisson distributed.
    sy = np.sqrt(counts[condition])

    chi2fit = Chi2Regression(double_gaussian, x, y, sy)

    minuit_chi2 = Minuit(
        chi2fit, N1=1000000, N2=1000000, mu1=0.0, mu2=0.0, sigma1=1, sigma2=5
    )

    minuit_chi2.migrad()

    # The fitted values of the parameters
    fit_N1, fit_N2, fit_mu1, fit_mu2, fit_sigma1, fit_sigma2 = minuit_chi2.values[:]  

    # Loop to get both parameter values and uncertainties:
    for name in minuit_chi2.parameters:
        value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
        print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")

    # Get Chi2 value; The value minimised, i.e. Chi2 or -2*LogLikeliHood (LLH) value
    chi2_value = (minuit_chi2.fval)

    # Get number of degrees-of-freedom (Ndof):
    N_NotEmptyBin = np.sum(y > 0)

    # ERDA version (older version of Minuit!)
    Ndof_value = N_NotEmptyBin - len(minuit_chi2.values)  

    # The chi2 probability given N_DOF degrees of freedom
    Prob_value = stats.chi2.sf(chi2_value, Ndof_value)  
    print(
        f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3e}"
    )

    # Adding fit results to plot:
    d = {
        "N1": [minuit_chi2.values["N1"], minuit_chi2.errors["N1"]],
        "mu1": [minuit_chi2.values["mu1"], minuit_chi2.errors["mu1"]],
        "sigma1": [minuit_chi2.values["sigma1"], minuit_chi2.errors["sigma1"]],
        "N2": [minuit_chi2.values["N2"], minuit_chi2.errors["N2"]],
        "mu2": [minuit_chi2.values["mu2"], minuit_chi2.errors["mu2"]],
        "sigma2": [minuit_chi2.values["sigma2"], minuit_chi2.errors["sigma2"]],
        "Chi2": chi2_value,
        "ndf": Ndof_value,
        "Prob": Prob_value,
    }


    fig, ax = plt.subplots(figsize=(16, 8))  # figsize is in inches
    ax.errorbar(
        x,
        y,
        yerr=sy,
        xerr=0.0,
        label="Data, with Poisson errors",
        fmt=".k",
        ecolor="k",
        elinewidth=1,
        capsize=1,
        capthick=1,
    )

    xmin, xmax = -10, 10
    ax.set_xlim(xmin, xmax)
    x_axis = np.linspace(xmin - 0.001, xmax + 0.001, 1000)
    ax.plot(
        x_axis,
        double_gaussian(x_axis, *minuit_chi2.values),
        "-r",
        label="Chi2 fit double gaussian distribution",
    )
    ax.plot(
        x_axis,
        gaussian(x_axis, fit_N1, fit_mu1, fit_sigma1),
        "-b",
        label="Chi2 fit gaussian 1 distribution",
    )
    ax.plot(
        x_axis,
        gaussian(x_axis, fit_N2, fit_mu2, fit_sigma2),
        "-g",
        label="Chi2 fit gaussian 2 distribution",
    )

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.62, 0.90, text, ax, fontsize=16)

    ax.set(
        xlabel="Z-score",  # the label of the y axis
        ylabel="Frequency",  # the label of the y axis
        title=f"Distribution of z-scores for {pattern}",
    )  # the title of the plot
    ax.legend(loc="upper left")
    # could also be # loc = 'upper right' e.g.
    plt.savefig(args.output + pattern +"_Z_score.png"    )
