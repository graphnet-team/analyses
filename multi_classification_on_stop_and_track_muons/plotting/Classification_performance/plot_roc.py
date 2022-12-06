import os, sys
# ability to fetch scripts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
from helper_functions.parsing import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from numpy import interp
import torch
from torch.nn.functional import one_hot

from sklearn.metrics import roc_curve, auc, roc_auc_score

def plot_roc(args):
    results = pd.read_csv(args.MC_csv_path, index_col=[0]).sort_values('event_no').reset_index()

    class_options = {1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2,}
    pid_transform = torch.tensor([class_options[int(value)] for value in results["pid"]])
    
    y_test = one_hot(pid_transform)
    y_prob = results[["pid_noise_pred","pid_muon_pred","pid_neutrino_pred"]]
    print("same shape: ", y_test.shape == y_prob.shape, y_test.shape)

    nb_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob.iloc[:, i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_prob.values.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    

    macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
    weighted_roc_auc_ovo = roc_auc_score(
        y_test, y_prob, multi_class="ovo", average="weighted"
    )
    macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
    weighted_roc_auc_ovr = roc_auc_score(
        y_test, y_prob, multi_class="ovr", average="weighted"
    )

    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=[20, 10])
    # inset axes
    axins = ax.inset_axes([0.56, 0.03, 0.42, 0.42])

    ax.plot(
        fpr["micro"], tpr["micro"],
        color="tab:olive", linestyle='dashdot',
        label='micro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["micro"]),
        linewidth=2)

    axins.plot(
        fpr["micro"], tpr["micro"],
        color="tab:olive", linestyle='dashdot',
        label='micro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["micro"]),
        linewidth=2)

    ax.plot(
        fpr["macro"], tpr["macro"],
        color="tab:purple", linestyle='dashdot',
        label='macro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["macro"]),
        linewidth=2)

    axins.plot(
        fpr["macro"], tpr["macro"],
        color="tab:purple", linestyle='dashdot',
        label='macro-average ROC curve (area = {0:0.4f})'
                ''.format(roc_auc["macro"]),
        linewidth=2)


    colors=["tab:blue","tab:green","tab:orange"]


    for i, item in enumerate(["noise", "muon", "neutrino"]):
        ax.plot(
                fpr[i], tpr[i], 
                color=colors[i], label='ROC of {0} class (area = {1:0.4f})'
                    ''.format(item, roc_auc[i]))
        axins.plot(
                fpr[i], tpr[i], 
                color=colors[i], label='ROC of {0} class (area = {1:0.4f})'
                    ''.format(item, roc_auc[i]))


    ax.plot([0, 1], [0, 1], 'k--', label="random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic to multi-class')
    ax.legend(loc="upper right", framealpha=0.99)

    # sub region of the original image
    x1, x2, y1, y2 = 0, .02, 0.92, 1
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    ax.indicate_inset_zoom(axins, edgecolor="black")

    ax.text(
        .775, .6, 
        f"""One-vs-One ROC AUC scores:
        \n{macro_roc_auc_ovo:.4f} (macro), 
        \n{weighted_roc_auc_ovo:.4f} (weighted by prevalence)
        \nOne-vs-Rest ROC AUC scores:
        \n{macro_roc_auc_ovr:.4f} (macro), 
        \n{weighted_roc_auc_ovr:.4f} (weighted by prevalence)""" ,
        fontsize = 10, 
        bbox = dict(facecolor = 'white', alpha = 0.99))
    
    # ensure output exists
    os.makedirs(args.output, exist_ok=True)
    fig.savefig(args.output + 'Roc_curves.png')


def main(args):
    plot_roc(args)


if __name__ == "__main__":
    main(get_plotArgs())