import os, sys
# ability to fetch scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
from helper_functions.parsing import *
from helper_functions.plot_params import *

import numpy as np
import pandas as pd
import torch

def plot_probability_heatmaps(datapath, args):
    results = pd.read_csv(datapath).sort_values('event_no').reset_index(drop = True)

    class_options = {1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2,}
    pid_transform = torch.tensor([class_options[int(value)] for value in results["pid"]])
    truth = pid_transform

    number = len(results)

    mask_noise = truth == 0
    mask_muon = truth == 1
    mask_neutrino = truth == 2

    bins = 50

    fig, axs = plt.subplots(3,1,sharex=False,figsize=(8, 16))

    cd1 = axs[0].hist2d(
        results['pid_noise_pred'].values[:number][mask_noise],
        results['pid_neutrino_pred'].values[:number][mask_noise],
        bins,norm=colors.LogNorm()
        )
    cd2 = axs[1].hist2d(
        results['pid_noise_pred'].values[:number][mask_muon],
        results['pid_neutrino_pred'].values[:number][mask_muon],
        bins,norm=colors.LogNorm()
        )
    cd3 = axs[2].hist2d(
        results['pid_noise_pred'].values[:number][mask_neutrino],
        results['pid_neutrino_pred'].values[:number][mask_neutrino],
        bins,norm=colors.LogNorm()
        )

    axs[0].set_ylabel('Noise probability')
    axs[1].set_ylabel('Muon probability')
    axs[2].set_ylabel('Neutrino probability')

    axs[0].set_xlabel('Noise probability')
    axs[1].set_xlabel('Muon probability')
    axs[2].set_xlabel('Neutrino probability')

    axs[0].set_title('Noise')
    axs[1].set_title('Muons')
    axs[2].set_title('Neutrinos')

    fig.colorbar(cd1[3], ax=axs[0])
    fig.colorbar(cd2[3], ax=axs[1])
    fig.colorbar(cd3[3], ax=axs[2])

    fig.tight_layout()
    # ensure output exists
    os.makedirs(args.output, exist_ok=True)
    if "MC" in datapath.split("/")[-1][:-4]:
        fig.savefig(args.output + 'MC_probability_heatmaps.png')
    elif "RD" in datapath.split("/")[-1][:-4]:
        fig.savefig(args.output + 'RD_probability_heatmaps.png')
    else:
        fig.savefig(args.output + 'probability_heatmaps.png')

def main(args):
    if args.MC_csv_path is not None:
        plot_probability_heatmaps(datapath=args.MC_csv_path, args=args)
    # if args.RD_csv_path is not None:
    #     plot_probability_heatmaps(datapath=args.RD_csv_path, args=args)

if __name__ == "__main__":
    main(get_plotArgs())