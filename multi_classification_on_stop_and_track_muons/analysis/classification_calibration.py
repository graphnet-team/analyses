"""Example of non-parametric isotonic calibration of a classification Model."""

import os
from typing import cast
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

from graphnet.training.loss_functions import NLLLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_desired_event_numbers,
)
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.classification import ClassificationTask
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_train_validation_dataloader,
    make_dataloaders,
    save_results,
)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

# from graphnet.utilities.logging import get_logger

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


def main() -> None:
    """Run example."""
    # logger.info(f"features: {features}")
    # logger.info(f"truth: {truth}")

    # single integer definition of class
    # class_options = 3
    # list of classes
    # class_options = [0,1,2]
    # transformation of target to a given class integer
    class_options = {
        1: 0,
        -1: 0,
        13: 1,
        -13: 1,
        12: 2,
        -12: 2,
        14: 2,
        -14: 2,
        16: 2,
        -16: 2,
    }

    # Configuration
    config = {
        "db": "/groups/icecube/petersen/GraphNetDatabaseRepository/Leon_MC_data/last_one_lvl3MC.db",
        "pulsemap": "SplitInIcePulses",
        "batch_size": 512,
        "num_workers": 10,
        "accelerator": "cpu",  # gpu
        "devices": 1,  # [0],
        "target": "pid",
        "classification": class_options,
        "n_epochs": 10,
        "patience": 5,
    }
    archive = "/groups/icecube/petersen/GraphNetDatabaseRepository/example_results/train_classification_model"
    run_name = "dynedge_{}_example".format(config["target"])

    # Log configuration to W&B
    # wandb_logger.experiment.config.update(config)

    # Common variables
    train_selection = get_desired_event_numbers(
        database=cast(str, config["db"]),
        desired_size=1000,
        fraction_noise=float(1 / 3),
        fraction_muon=float(1 / 3),
        fraction_nu_e=float(1 / 9),
        fraction_nu_mu=float(1 / 9),
        fraction_nu_tau=float(1 / 9),
    )

    (
        training_dataloader,
        validation_dataloader,
        test_dataloader
    ) = make_dataloaders(
        data_path=cast(str, config["db"]),
        selection=train_selection,
        pulsemaps=cast(str, config["pulsemap"]),
        features=features,
        truth=truth,
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
        nb_classes=len(np.unique(list(class_options.values()))),
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=NLLLoss(options=config["classification"]),
    )
    model = StandardModel(
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
                len(training_dataloader) * cast(int, config["n_epochs"]),
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
        # logger=wandb_logger,
    )

    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        # logger.warning("[ctrl+c] Exiting gracefully.")
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

    #save_results(cast(str, config["db"]), run_name, results, archive, model)

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.cm.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i, (clf, name) in enumerate(clf_list):
        clf.fit(X_train, y_train)
        display = CalibrationDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots (Naive Bayes)")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (_, name) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
