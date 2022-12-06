import os, sys
# ability to fetch scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
from helper_functions.parsing import *
from helper_functions.plot_params import *

import pandas as pd
import numpy as np


def plot_PID_predictions(args) -> None:
    feature_data = pd.read_csv(args.MC_csv_path, index_col=[0])

    preds = ["pid_noise_pred","pid_muon_pred","pid_neutrino_pred"]
    pids = [1,12,13]

    ## transform data
    # remove anti
    feature_data['pid'] = feature_data['pid'].apply(abs)
    # make all neutrinos the same class; to match training classes
    feature_data.loc[feature_data['pid']==14, "pid"]=12
    feature_data.loc[feature_data['pid']==16, "pid"]=12

    preds = ["pid_noise_pred","pid_muon_pred","pid_neutrino_pred"]
    pids = [1,12,13]
    color=["b","y","g"]
    for i, pred in enumerate(preds):
        pid_list=[]
        pid_bin_centers=[]
        
        fig,ax = plt.subplots(figsize=(20,10))

        temp = 0
        for i, pid in enumerate(pids):
            y, bin_edges = np.histogram(feature_data[pred].loc[feature_data["pid"]==pid], bins=50)
            
            pid_list.append(feature_data[pred].loc[feature_data["pid"]==pid])
            pid_bin_centers.append(0.5*(bin_edges[1:] + bin_edges[:-1]))
        
        ax.hist(pid_list, bins=100, log=True, stacked=True, label=("noise", "neutrino", "muon"))
        ax.set(yscale='log')
        ax.set_xlim(-0.001,1.001)
        #ax.set(ylabel='Rate',yscale='log')
        #ax.yaxis.set_major_formatter(ticker.EngFormatter(unit='Hz'))
        #for i, _ in enumerate(pids):
        #    plt.errorbar(pid_bin_centers[i], pid_histlist[i], yerr = stds[i], fmt=f".{color[i]}")
        plt.title(pred.replace("_"," ")[4:])
        plt.legend()
        # ensure output exists
        os.makedirs(args.output, exist_ok=True)
        plt.savefig(args.output + pred.replace("_","")[3:-4]+".png")


def main(args):
    plot_PID_predictions(args)

if __name__ == "__main__":
    main(get_plotArgs())

