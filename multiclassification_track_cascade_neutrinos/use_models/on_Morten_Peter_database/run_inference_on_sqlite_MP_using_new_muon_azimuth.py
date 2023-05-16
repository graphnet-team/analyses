"""Running inference on GraphSAGE-cleaned pulses in IceCube-Upgrade."""

import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
from torch.nn.functional import one_hot, softmax


from graphnet.training.loss_functions import CrossEntropyLoss,BinaryCrossEntropyLoss, LogCoshLoss,VonMisesFisher2DLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_desired_event_numbers,
    get_equal_proportion_neutrino_indices,
)
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import MulticlassClassificationTask, BinaryClassificationTask, EnergyReconstruction, ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    make_dataloader,
    save_results,
)
from graphnet.utilities.logging import get_logger

import numpy as np
import pandas as pd
import csv

# logger = get_logger(logging.DEBUG)

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Initialise Weights & Biases (W&B) run
# wandb_logger = WandbLogger(
#    project="example-script",
#    entity="graphnet-team",
#    save_dir="./wandb/",
#    log_model=True,
# )

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

# Main function definition
def main(
    input_path: str,
    output_path: str,
    model_path: str,
):
    #test_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_new_muons_peter/train_val_test_split/Multiclassification/Multiclassification_test_event_no.csv').reset_index(drop = True)['event_no'].ravel().tolist()
    #MC_selection = pd.read_csv('/groups/icecube/peter/workspace/analyses/multi_classification_on_stop_and_track_muons/plotting/Comparison_RD_MC/dev_lvl3_genie_burnsample_MC_event_numbers.csv').reset_index(drop = True)['event_no'].ravel().tolist()
    #test_selection = test_selection[:1000]
    # Configuration
    config = {
        "db": input_path,
        "pulsemap": "SplitInIcePulses",
        "batch_size": 512,
        "num_workers": 20,
        "accelerator": "gpu",
        "devices": [1],
        "target": "azimuth",
        "n_epochs": 1,
        "patience": 1,
    }
    archive = output_path
    run_name = "dynedge_trained_on_Peter_Morten_RD_{}_predict_azimuth_MP_set".format(
        config["target"]
    )
    config['run_name'] = run_name

    # Log configuration to W&B
    # wandb_logger.experiment.config.update(config)
    # train_selection, _ = get_equal_proportion_neutrino_indices(config["db"])
    # train_selection = train_selection[0:50000]

#    prediction_dataloader_MC = make_dataloader(
    #    config["db"],
    #    config["pulsemap"],
    #    features,
    #    truth,
    #    selection = MC_selection,
    #    batch_size=config["batch_size"],
    #    shuffle=False,
    #    num_workers=config["num_workers"],
#    )

    prediction_dataloader_test = make_dataloader(
        config["db"],
        config["pulsemap"],
        features,
        truth,
        #selection = test_selection,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # Building model
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        nb_neighbours = 8,
        global_pooling_schemes = ['min', 'max', 'mean','sum'],
        add_global_variables_after_pooling=True,
    )
    
    if config["target"] =='zenith':
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher2DLoss(),
        )

    elif config["target"] == 'azimuth':
        task = AzimuthReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher2DLoss(),
        )

    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        default_root_dir=f'~/{config["run_name"]}',
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        #logger=wandb_logger,
    )

    # Load model
    model.load_state_dict(model_path)

#    # Saving predictions to file
#    resultsMC = get_predictions(
#        trainer,
#        model,
#        prediction_dataloader_MC,
#        [config["target"] + "_noise_pred", config["target"] + "_muon_pred", config["target"]+ "_neutrino_pred"],
#        additional_attributes=[config["target"], "event_no"],
#    )
#
#    # save_results(config["db"], run_name, results, archive, model)
#    resultsMC.to_csv(
#        output_folder + "/{}_Burnsample_MC_Full_db.csv".format(config["target"])
#    )


    # Saving predictions to file
    resultstest = get_predictions(
        trainer,
        model,
        prediction_dataloader_test,
        [config["target"] + "_pred", config["target"] + "_kappa"],
        additional_attributes=["event_no"],
    )

    # save_results(config["db"], run_name, results, archive, model)
    resultstest.to_csv(
        output_folder + "/{}_Azimuth_Whole_MP.csv".format(config["target"])
    )

# Main function call
if __name__ == "__main__":

    input_db = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/merged_database/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003.db"
    output_folder = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_new_muons_Peter_database/inference/old_morten_peter_db"
    model_path = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_track_cascade_neutrino/using_MP_lvl3/trained_models/osc_next_level3_v2/MP_data_azimuth_test_1_mill_attempt2_test_set_equal_track_cascade/MP_data_azimuth_test_1_mill_attempt2_test_set_state_dict.pth"

    main(input_db, output_folder, model_path)
