import os, sys
# ability to fetch scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
from helper_functions.parsing import *
from helper_functions.plot_params import *

import numpy as np
import pandas as pd
from sklearn import metrics
import torch

def plot_cm(args):
    results = pd.read_csv(args.MC_csv_path).sort_values('event_no').reset_index(drop = True)

    class_options = {1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2,}
    pid_transform = torch.tensor([class_options[int(value)] for value in results["pid"]])
    truth = pid_transform

    # return series of row wise max
    predictions = results[['pid_noise_pred','pid_muon_pred','pid_neutrino_pred']].idxmax(axis="columns")
    predictions[predictions == "pid_noise_pred"] = 0
    predictions[predictions == "pid_muon_pred"] = 1
    predictions[predictions == "pid_neutrino_pred"] = 2
    
    noise_number = len(truth[truth == 0] == True)
    muon_number = len(truth[truth == 1] == True)
    neutrino_number = len(truth[truth == 2] == True)

    print(f'there are {noise_number} noise, {muon_number} muons, and {neutrino_number} neutrinos')

    confusion_matrix = metrics.confusion_matrix(truth.tolist(), predictions.values.tolist())
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Noise','Muons','Neutrinos'])
    cm_display.plot()
    
    plt.savefig(args.output + 'Confusion_matrix.png')

    
def main(args):
    plot_cm(args)


if __name__ == "__main__":
    main(get_plotArgs())