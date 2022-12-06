"""Example of training Model."""

import os
from typing import cast, Dict, Any
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
from torch.nn.functional import softmax

from graphnet.training.loss_functions import CrossEntropyLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_desired_event_numbers,
)
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.classification import ClassificationTask
from graphnet.training.callbacks import ProgressBar
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    save_results,
)

from graphnet.utilities.logging import get_logger

import ast
import argparse

logger = get_logger()

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

# Initialise Weights & Biases (W&B) run
wandb_logger = WandbLogger(
    project="Classification",
    entity="graphnet-team",
    save_dir=WANDB_DIR,
    log_model=True,
)

import argparse
print('WandB initialized')

parser = argparse.ArgumentParser(
    description="plotting the predicted zenith and azimuth vs truth."
)
parser.add_argument(
    "-db",
    "--database",
    dest="path_to_db",
    type=str,
    help="<required> path(s) to database [list]",
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/Leon_MC_data/last_one_lvl3MC.db",
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="<required> the output path [str]",
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Datasets/last_one_lvl3MC.db" ,
)
parser.add_argument(
    "-c",
    "--class_options",
    dest="class_options",
    type=str,
    help="class options as a dict conversion. [str]",
    default='{1: 0, -1: 0, 13: 1, -13: 1, 12: 2, -12: 2, 14: 2, -14: 2, 16: 2, -16: 2}',
)
parser.add_argument(
    "-t",
    "--target",
    dest="target",
    type=str,
    help="target for training on. [str]",
    default='pid',
)
parser.add_argument(
    "-n",
    "--event_numbers",
    dest="event_numbers",
    type=int,
    help="the number of muons to train on; if too high will take all available. [int]",
    default=5000,
)
parser.add_argument(
    "-p",
    "--pulsemap",
    dest="pulsemap",
    type=str,
    help="<required> the pulsemap to use. [str]",
    default="SplitInIcePulses",
)
parser.add_argument(
    "-a",
    "--accelerator",
    dest="accelerator",
    type=str,
    help="<required> the name for the model. [str]",
    default="gpu"
)
parser.add_argument(
    "-g",
    "--gpu_number",
    dest="gpu_number",
    type=int,
    help="<required> the name for the model. [str]",
    default=0
)
parser.add_argument(
    "-b",
    "--batch_size",
    dest="batch_size",
    type=int,
    help="<required> the name for the model. [str]",
    default=512,
)
parser.add_argument(
    "-e",
    "--epochs",
    dest="epochs",
    type=int,
    help="<required> number of epochs. [int]",
    default=10,
)
parser.add_argument(
    "-s",
    "--patience",
    dest="patience",
    type=int,
    help="<required> number of epochs after early stopping is triggered. [int]",
    default=10,
)
parser.add_argument(
    "-w",
    "--workers",
    dest="workers",
    type=int,
    help="<required> the number of cpu's to use. [int]",
    default=10,
)
parser.add_argument(
    "-r",
    "--run_name",
    dest="run_name",
    type=str,
    help="<required> the name for the model. [str]",
    default="debug_leonMC_5k"
)


args = parser.parse_args()

def train(config: Dict[str, Any]) -> None:
    """Train model with configuration given by `config`."""
    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    # Common variables; equal distribution of pid.
    train_selection = get_desired_event_numbers(
        database=cast(str, config["db"]),
        desired_size=cast(int, config["event_numbers"]),
        fraction_noise=float(1 / 3),
        fraction_muon=float(1 / 3),
        fraction_nu_e=float(1 / 9),
        fraction_nu_mu=float(1 / 9),
        fraction_nu_tau=float(1 / 9),
    )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    (
        training_dataloader,
        validation_dataloader,
    ) = make_train_validation_dataloader(
        cast(str, config["db"]),
        train_selection,
        cast(str, config["pulsemap"]),
        features,
        truth,
        batch_size=cast(int, config["batch_size"]),
        num_workers=cast(int, config["num_workers"]),
    )

    # Building model
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = ClassificationTask(
        nb_classes=len(np.unique(list(config["class_options"].values()))),
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=CrossEntropyLoss(options=config["class_options"]),
        transform_inference=lambda x: softmax(x, dim=-1),
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-04, "eps": 1e-03},
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
        logger=wandb_logger,
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        logger.warning("[ctrl+c] Exiting gracefully.")
        pass

    # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        validation_dataloader,
        [
            cast(str, config["target"]) + "_noise_pred",
            cast(str, config["target"]) + "_muon_pred",
            cast(str, config["target"]) + "_neutrino_pred",
        ],
        additional_attributes=[cast(str, config["target"]), "event_no"],
    )

    save_results(
        cast(str, config["db"]),
        config["run_name"],
        results,
        config["archive"],
        model,
    )


def main() -> None:
    """Run example."""
    # transformation of target to a given class integer
    
    target = args.target
    run_name = "dynedge_{}_{}".format(target, args.run_name)

    if "cpu" in args.accelerator:
        device = 1
    elif "gpu" in args.accelerator:
        device = [args.gpu_number]
    else:
        print('Use correct accelerator; "cpu" or "gpu"')
        exit

    # Configuration
    config = {
        "db": args.path_to_db,
        "pulsemap": args.pulsemap,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "accelerator": args.accelerator,
        "devices": device,
        "target": target,
        "event_numbers": args.event_numbers,
        "class_options": ast.literal_eval(args.class_options),
        "n_epochs": args.epochs,
        "patience": args.patience,
        "archive": args.output,
        "run_name": run_name,
    }

    train(config)


if __name__ == "__main__":
    main()
