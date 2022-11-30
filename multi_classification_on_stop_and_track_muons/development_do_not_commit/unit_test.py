from typing import cast
import numpy as np

from graphnet.models.task.classification import ClassificationTask
from graphnet.training.loss_functions import NLLLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.training.utils import (
    make_train_validation_dataloader,
)
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.data.sqlite.sqlite_selection import (
    get_desired_event_numbers,
)

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]

class_options = {
    1:0,-1:0,
    13:1,-13:1,
    12:2,-12:2,14:2,-14:2,16:2,-16:2
    }

config = {
        "db": "/groups/icecube/petersen/GraphNetDatabaseRepository/Leon_MC_data/last_one_lvl3MC.db",
        "pulsemap": "SplitInIcePulses",
        "batch_size": 512,
        "num_workers": 10,
        "target": "pid",
        "classification": class_options,
    }
import random
pid_sample = [1,-1,12,-12,13,-13,14,-14,16,-16]
pid_list = random.choices(pid_sample,k=500)
train_selection = pid_list

# Common variables
train_selection = get_desired_event_numbers(
    database = cast(str, config["db"]),
    desired_size = 1000,
    fraction_noise = 0.2,
    fraction_muon = 0.2,
    fraction_nu_e = 0.2,
    fraction_nu_mu = 0.2,
    fraction_nu_tau = 0.2,
)


(training_dataloader, validation_dataloader,) = make_train_validation_dataloader(
    db = cast(str, config["db"]),
    selection = train_selection,
    pulsemaps= cast(str, config["pulsemap"]),
    features = features,
    truth = truth,
    batch_size = cast(int, config["batch_size"]),
    num_workers = cast(int, config["num_workers"]),
)

# Building model
detector = IceCubeDeepCore(
    graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
)
gnn = DynEdge(
    nb_inputs=detector.nb_outputs,
    global_pooling_schemes=["min", "max", "mean", "sum"],
)
ClassificationTask(
        nb_inputs=len(np.unique(list(class_options.values()))),
        hidden_size = 128,
        target_labels = config["target"],
        loss_function = NLLLoss(options=class_options),
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
    logger=wandb_logger,
)

try:
    trainer.fit(model, training_dataloader, validation_dataloader)
except KeyboardInterrupt:
    logger.warning("[ctrl+c] Exiting gracefully.")
    pass