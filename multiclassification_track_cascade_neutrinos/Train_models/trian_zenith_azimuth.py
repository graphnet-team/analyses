import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam
import pandas as pd
from graphnet.training.loss_functions import VonMisesFisher2DLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import (
    get_predictions,
    make_dataloader,
    save_results,
)
from graphnet.utilities.logging import get_logger

logger = get_logger()

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]
# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

import random

def train(config,wandb_logger):
    train_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/Selectionsneutrinos_train_from_logit_selection.csv').reset_index(drop = True)['event_no'].ravel().tolist()
    validation_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/Selectionsneutrinos_val_from_logit_selection.csv').reset_index(drop = True)['event_no'].ravel().tolist()
    test_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/Selectionsneutrinos_test_from_logit_selection.csv').reset_index(drop = True)['event_no'].ravel().tolist()
    random.shuffle(train_selection)
    random.shuffle(validation_selection)
    random.shuffle(test_selection)

    train_selection = train_selection[:1000000]
    validation_selection = validation_selection[:100000]
    test_selection = test_selection


    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)

    # Common variables
    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    training_dataloader = make_dataloader(db = config['db'],
                                            selection = train_selection, #config['train_selection'],
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = True)

    validation_dataloader = make_dataloader(db = config['db'],
                                            selection = validation_selection, #config["validation_selection"],
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False)

    test_dataloader = make_dataloader(db = config['db'],
                                            selection = test_selection, #config["test_selection"],
                                            pulsemaps = config['pulsemap'],
                                            features = features,
                                            truth = truth,
                                            batch_size = config['batch_size'],
                                            num_workers = config['num_workers'],
                                            shuffle = False)
    # Building model
    detector = IceCubeDeepCore(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        nb_neighbours = config["nb_neighbors"],
        global_pooling_schemes = ['min', 'max', 'mean'],
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
        default_root_dir=f'~/{config["run_name"]}',
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

    # Predict on Test Set and save results to file
    results = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = test_dataloader,
        prediction_columns =[config["target"] + "_pred", config["target"] + "_kappa"],
        additional_attributes=[config["target"], "event_no", "energy"],
    )
    save_results(config["db"], config["run_name"] + '_test_set', results, config["archive"], model)

    # Predict on Validation Set and save results to file
    results = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = validation_dataloader,
        prediction_columns =[config["target"] + "_pred", config["target"] + "_kappa"],
        additional_attributes=[config["target"], "event_no", "energy"],
    )
    save_results(config["db"], config["run_name"] + '_validation_set', results, config["archive"], model)

# Main function definition
def main():
    for target in ['azimuth']:#['zenith', 'azimuth']:


        pulsemap = 'SplitInIcePulses'
        nb_neighbours = 8
        n_epochs = 20
        archive = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track"
        run_name = f"Peter_Morten_{target}_test_1_mill"
        
        # Initialise Weights & Biases (W&B) run
        wandb_logger = WandbLogger(
        name=run_name,
        project="Test",
        entity="graphnet-team",
        save_dir=WANDB_DIR,
        log_model=True
        )
        # Selections
        #train_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/northeren_tracks/dev_northern_tracks_full_part_1/selections/benchmark_train_selection.csv').reset_index(drop = True)['event_no'].ravel().tolist()
        #validation_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/northeren_tracks/dev_northern_tracks_full_part_1/selections/benchmark_validate_selection.csv').reset_index(drop = True)['event_no'].ravel().tolist()
        #test_selection = pd.read_csv('/groups/icecube/petersen/GraphNetDatabaseRepository/northeren_tracks/dev_northern_tracks_full_part_1/selections/benchmark_test_selection.csv').reset_index(drop = True)['event_no'].ravel().tolist()

        # Configuration
        config = {
            "db": "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track.db",
            #"train_selection": train_selection,
            #"test_selection": test_selection,
            #"validation_selection": validation_selection,
            "pulsemap": pulsemap,
            "batch_size": 512,
            "num_workers": 10,
            "accelerator": "gpu",
            "devices": [0],
            "target": target,
            "n_epochs": n_epochs,
            "patience": 5,
            "archive": archive,
            "run_name": run_name,
            "nb_neighbors": nb_neighbours,
            }

        
        train(config,wandb_logger)
        wandb_logger.finalize("success")
# Main function call
if __name__ == "__main__":
    main()