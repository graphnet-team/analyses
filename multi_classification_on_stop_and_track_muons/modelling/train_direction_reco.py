import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.components.loss_functions import VonMisesFisher2DLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_equal_proportion_neutrino_indices,
    get_desired_event_numbers,
)
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn.dynedge import DynEdge, DOMCoarsenedDynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import (
    ZenithReconstructionWithKappa,
    AzimuthReconstructionWithKappa,
)
from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    save_results,
)
from graphnet.utilities.logging import get_logger

# logger = get_logger()

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

# Initialise Weights & Biases (W&B) run
# wandb_logger = WandbLogger(
#    project="example-script",
#    entity="graphnet-team",
#    save_dir=WANDB_DIR,
#    log_model=True,
# )
#


def train(config):
    # Log configuration to W&B
    #   wandb_logger.experiment.config.update(config)

    # Common variables
    # train_selection, _ = get_equal_proportion_neutrino_indices(config["db"])
    # train_selection = train_selection[0:]#config["max_events"]]
    train_selection = get_desired_event_numbers(
        config["db"], desired_size=50000, fraction_muon=1
    )
    #    logger.info(f"features: {features}")
    #    logger.info(f"truth: {truth}")

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        config["db"],
        train_selection,
        config["pulsemap"],
        features,
        truth,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # Building model
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    if config["node_pooling"]:
        gnn = DOMCoarsenedDynEdge(
            nb_inputs=detector.nb_outputs,
        )
    else:
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
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["n_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
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
        #        logger=wandb_logger,
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        #        logger.warning("[ctrl+c] Exiting gracefully.")
        pass

    # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [config["target"] + "_pred", config["target"] + "_kappa_pred"],
        additional_attributes=[config["target"], "event_no"],
    )

    save_results(
        config["db"], config["run_name"], results, config["archive"], model
    )


# Main function definition
def main():
    for target in ["zenith", "azimuth"]:
        archive = "/groups/icecube/qgf305/storage/MoonPointing/Models/MC/GenieL2_Muon_Leon"
        run_name = "dynedge_{}_GenieL2_Muon_Leon".format(target)

        # Configuration
        config = {
            "db": "/groups/icecube/leonbozi/work2/data/lvl3_dbs/last_one_lvl3MC/data/last_one_lvl3MC.db",
            "pulsemap": "SRTInIcePulses",
            "batch_size": 1024,
            "num_workers": 10,
            "accelerator": "gpu",
            "devices": [0],
            "target": target,
            "n_epochs": 20,
            "patience": 5,
            "archive": archive,
            "run_name": run_name,
            "max_events": 500000,
            "node_pooling": False,
        }
        train(config)


# Main function call
if __name__ == "__main__":
    main()
