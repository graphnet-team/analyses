import sqlite3 as sql
from plot_params import *
from iminuit import Minuit
import pandas as pd
from pandas import read_sql
import numpy as np
from scipy import stats
from scipy.stats import binom, poisson, norm
import sys

sys.path.append('/groups/icecube/peter/workspace/External_functions')
from Troels_external_functions import Chi2Regression
from Troels_external_functions import nice_string_output, add_text_to_ax   # Useful functions to print fit results on figure



bin_number = 500

zenith_db = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Trained_Models/dynedge_zenith_all_example/results.csv"
zenith_db = pd.read_csv(zenith_db)
print(zenith_db.head(10))

zenith, zenith_pred,  = zenith_db.zenith, zenith_db.zenith_pred
zenith_std = 1/np.sqrt(zenith_db.zenith_kappa_pred)
zenith_diff = zenith_pred - zenith
zenith_z = zenith_diff/zenith_std


def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)



def double_gaussian(x, N1,N2, mu1,mu2, sigma1,sigma2):
    return N1 * 1.0 / (sigma1*np.sqrt(2*np.pi)) * np.exp(-0.5* (x-mu1)**2/sigma1**2) +  N2 * 1.0 / (sigma2*np.sqrt(2*np.pi)) * np.exp(-0.5* (x-mu2)**2/sigma2**2)

def gaussian(x, N, mu,sigma):
    return N * 1.0 / (sigma*np.sqrt(2*np.pi)) * np.exp(-0.5* (x-mu)**2/sigma**2)


###

#std_pred10 = np.std(zenith_z[np.abs(zenith_z) < 10])
#print(std_pred10)
#def Gauss(x, mu, sig):
#    return 1/(sig*np.sqrt(2*np.pi))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

#x = np.linspace(-10,10,1000)
#y = Gauss(x,0,1.0)
#
#plt.figure()
#plt.hist(zenith_z, bins = bin_number,density=True)
#plt.plot(x,y)
#plt.xlim([-10,10])
#plt.title("results: zenith Z-score")
#plt.legend()
target = "zenith"
xmin,xmax = -10,10

counts, bin_edges = np.histogram(zenith_z, bins=bin_number)
x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
y = counts[counts>0]
sy = np.sqrt(counts[counts>0])         # NOTE: We (naturally) assume that the bin count is Poisson distributed.

chi2fit = Chi2Regression(double_gaussian, x, y, sy)

minuit_chi2 = Minuit(chi2fit, N1=1000000,N2=1000000, mu1=0.0,mu2=0.0, sigma1=1,sigma2=5)

minuit_chi2.migrad(); 

fit_N1,fit_N2, fit_mu1,fit_mu2, fit_sigma1,fit_sigma2 = minuit_chi2.values[:]   # The fitted values of the parameters

# Loop to get both parameter values and uncertainties:
for name in minuit_chi2.parameters:
    value, error = minuit_chi2.values[name], minuit_chi2.errors[name]
    print(f"Fit value: {name} = {value:.5f} +/- {error:.5f}")

# Get Chi2 value:
chi2_value = minuit_chi2.fval            # The value minimised, i.e. Chi2 or -2*LogLikeliHood (LLH) value

# Get number of degrees-of-freedom (Ndof):
N_NotEmptyBin = np.sum(y > 0)

Ndof_value = N_NotEmptyBin - len(minuit_chi2.values)    # ERDA version (older version of Minuit!)

Prob_value = stats.chi2.sf(chi2_value, Ndof_value) # The chi2 probability given N_DOF degrees of freedom
print(f"Chi2 value: {chi2_value:.1f}   Ndof = {Ndof_value:.0f}    Prob(Chi2,Ndof) = {Prob_value:5.3e}")

# Adding fit results to plot:
d = {'N1':   [minuit_chi2.values['N1'], minuit_chi2.errors['N1']],
     'mu1':       [minuit_chi2.values['mu1'], minuit_chi2.errors['mu1']],
     'sigma1':       [minuit_chi2.values['sigma1'], minuit_chi2.errors['sigma1']],
     'N2':   [minuit_chi2.values['N2'], minuit_chi2.errors['N2']],
     'mu2':       [minuit_chi2.values['mu2'], minuit_chi2.errors['mu2']],
     'sigma2':       [minuit_chi2.values['sigma2'], minuit_chi2.errors['sigma2']],
     'Chi2':     chi2_value,
     'ndf':      Ndof_value,
     'Prob':     Prob_value,
    }



fig, ax = plt.subplots(figsize=(16, 8))  # figsize is in inches
ax.errorbar(x, y, yerr=sy, xerr=0.0, label='Data, with Poisson errors', fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)
ax.set_xlim(xmin,xmax)
x_axis = np.linspace(xmin-0.001, xmax+0.001, 1000)
ax.plot(x_axis, double_gaussian(x_axis, *minuit_chi2.values), '-r', label='Chi2 fit double gaussian distribution') 
ax.plot(x_axis, gaussian(x_axis, fit_N1,fit_mu1,fit_sigma1), '-b', label='Chi2 fit gaussian 1 distribution')
ax.plot(x_axis, gaussian(x_axis, fit_N2,fit_mu2,fit_sigma2), '-g', label='Chi2 fit gaussian 2 distribution') 

text = nice_string_output(d, extra_spacing=2, decimals=3)
add_text_to_ax(0.62, 0.90, text, ax, fontsize=16)

ax.set(xlabel="Z-score", # the label of the y axis
       ylabel="Frequency",  # the label of the y axis
       title=f"Distribution of z-scores for {target}") # the title of the plot
ax.legend(loc='upper left'); # could also be # loc = 'upper right' e.g.
plt.savefig("/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/plotting/Test_Plots/Model_trained_on_neutrinos/Uncertainty_test/zenith_Z_score.png")



