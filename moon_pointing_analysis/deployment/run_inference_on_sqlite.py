"""Running inference on GraphSAGE-cleaned pulses in IceCube-Upgrade."""

import logging

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
#from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import LogCoshLoss, VonMisesFisher2DLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_equal_proportion_neutrino_indices,
)
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import (
    get_predictions, 
    make_dataloader, #make_train_validation_dataloader
    save_results,
)
from graphnet.utilities.logging import get_logger

#logger = get_logger(logging.DEBUG)

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Initialise Weights & Biases (W&B) run
#wandb_logger = WandbLogger(
#    project="example-script",
#    entity="graphnet-team",
#    save_dir="./wandb/",
#    log_model=True,
#)

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

# Main function definition
def main(
    input_path: str,
    output_path: str,
    model_path: str,
):

    # Configuration
    config = {
        "db": input_path,
        "pulsemap": "InIceDSTPulses",
        "batch_size": 512,
        "num_workers": 10,
        "accelerator": "gpu",
        "devices": [0],
        "target": "zenith",
        "n_epochs": 1,
        "patience": 1,
    }
    archive = output_path
    run_name = "dynedge_trained_on_leon_MC_{}_predict_azimuth".format(config["target"])

    # Log configuration to W&B
    #wandb_logger.experiment.config.update(config)
    #train_selection, _ = get_equal_proportion_neutrino_indices(config["db"])
    #train_selection = train_selection[0:50000]

    prediction_dataloader = make_dataloader(
        config["db"],
        config["pulsemap"],
        features,
        truth,
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
    )
    if config["target"] == "zenith":
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher2DLoss(),
        )
    elif config["target"] == "azimuth":
        task = AzimuthReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher2DLoss(),
        )

    model = Model(
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
        accelerator=config["accelerator"],
        devices=config["devices"],
        max_epochs=config["n_epochs"],
        callbacks=callbacks,
        log_every_n_steps=1,
        #logger=wandb_logger,
    )

    # Load model
    model.load_state_dict(model_path)

    # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        prediction_dataloader,
        [config["target"] + "_pred",config["target"] + "_kappa_pred" ],
        additional_attributes=[config["target"], "event_no"],
    )

    #save_results(config["db"], run_name, results, archive, model)
    results.to_csv(output_folder + "/{}_Leon_MC_results.csv".format(config["target"]))

# Main function call
if __name__ == "__main__":

    input_db = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database/Merged_database.db"
    output_folder = "/groups/icecube/peter/storage/MoonPointing/data/Sschindler_data_L4/Merged_database"
    model_path = "/groups/icecube/peter/storage/MoonPointing/Models/Leon_Muon_data_MC/last_one_lvl3MC/dynedge_zenith_Leon_muon_data_MC/dynedge_zenith_Leon_muon_data_MC_state_dict.pth"

    main(input_db, output_folder, model_path)
