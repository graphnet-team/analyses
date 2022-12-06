import os, sys
# ability to fetch scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
from helper_functions.parsing import *
from helper_functions.plot_params import *
from helper_functions.external_functions import logit_transform

import pandas as pd
import scipy.optimize as optimize
import swifter

def plot_PID_predictions_logit(args):
    results_RD = pd.read_csv(args.RD_csv_path, index_col=[0])
    results_MC = pd.read_csv(args.MC_csv_path, index_col=[0])

    pid_transform = {1:0,12:2,13:1,14:2,16:2}

    truth_MC = []

    for i in range(len(results_MC)):
        truth_MC.append(pid_transform[abs(results_MC['pid'].values[i])])

    logits = dict()
    for item in ["noise", "muon", "neutrino"]:
        logits[item] = results_MC[f"pid_{item}_pred"].swifter.apply( lambda x: logit_transform(x,inverse=False) )
    logits = pd.DataFrame.from_dict(logits)

    fix, ax = plt.subplots(figsize=[10, 5])
    for item in ["noise", "muon", "neutrino"]:
        ax.hist(logits[item], bins=50, alpha=0.33, label=item);
    ax.set(yscale="log")
    ax.legend();
    # ensure output exists
    os.makedirs(args.output, exist_ok=True)
    plt.savefig(args.output + "logits_unstacked.png")

def main(args):
    plot_PID_predictions_logit(args)

if __name__ == "__main__":
    main(get_plotArgs())