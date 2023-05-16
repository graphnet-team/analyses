from distutils.log import debug
import logging
import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.optim.adam import Adam
from torch.nn.functional import one_hot, softmax


from graphnet.training.loss_functions import CrossEntropyLoss, BinaryCrossEntropyLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.data.sqlite.sqlite_selection import (
    get_desired_event_numbers,
    get_equal_proportion_neutrino_indices,
)
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import MulticlassClassificationTask,BinaryClassificationTask
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

print('All is imported')

#def multiclass_transform(target, num_classes=3):
#    pid_transform = {1:0,12:2,13:1,14:2,16:2}
#    return one_hot(torch.tensor([pid_transform[np.abs(value)] for value in target]), num_classes)

logger = get_logger()
# set increased verbose information when debugging.
logger.setLevel(logging.DEBUG)

# Configurations
torch.multiprocessing.set_sharing_strategy("file_system")

# Constants
features = FEATURES.DEEPCORE
truth = TRUTH.DEEPCORE[:-1]
#truth.append('track_mu')

# Make sure W&B output directory exists
WANDB_DIR = "./wandb/"
os.makedirs(WANDB_DIR, exist_ok=True)

# Initialise Weights & Biases (W&B) run
wandb_logger = WandbLogger(
    project="example-script",
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
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track.db",
    # required=True,
)
parser.add_argument(
    "-o",
    "--output",
    dest="output",
    type=str,
    help="<required> the output path [str]",
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models" ,
    # required=True,
)
parser.add_argument(
    "-p",
    "--pulsemap",
    dest="pulsemap",
    type=str,
    help="<required> the pulsemap to use. [str]",
    default="SplitInIcePulses",
    # required=True,
)
parser.add_argument(
    "-n",
    "--event_numbers",
    dest="event_numbers",
    type=int,
    help="the number of muons to train on; if too high will take all available. [int]",
    default=1000#int(7500000*3),
)
parser.add_argument(
    "-g",
    "--gpu",
    dest="gpu",
    type=int,
    help="<required> the name for the model. [str]",
    default=1# required=True,
)
parser.add_argument(
    "-b",
    "--batch_size",
    dest="batch_size",
    type=int,
    help="<required> the name for the model. [str]",
    default=512,
    # required=True,
)
parser.add_argument(
    "-e",
    "--epochs",
    dest="epochs",
    type=int,
    help="<required> the name for the model. [str]",
    default=20,
    # required=True,
)
parser.add_argument(
    "-w",
    "--workers",
    dest="workers",
    type=int,
    help="<required> the number of cpu's to use. [str]",
    default=25,
    # required=True,
)
parser.add_argument(
    "-r",
    "--run_name",
    dest="run_name",
    type=str,
    help="<required> the name for the model. [str]",
    default="Track_cascade_osc_next_level3_v2.00_genie_muongun_noise_120000_140000_160000_130000_888003_track_SplitInIcePulses_on_all_neutrinos"
    # required=True,
)
parser.add_argument(
    "-a",
    "--accelerator",
    dest="accelerator",
    type=str,
    help="<required> the name for the model. [str]",
    default="gpu"
    # required=True,
)

args = parser.parse_args()



print('Argparse done, defining main loop')
# Main function definition
def main():

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config = {
        "db": args.path_to_db,
        "pulsemap": args.pulsemap,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "accelerator": args.accelerator,
        "devices": [args.gpu],
        "target": "track_mu",
        "n_epochs": args.epochs,
        "patience": 5,
    }
    archive = args.output
    run_name = "dynedge_{}_".format(config["target"]) + args.run_name
    print('before logs to wand')
    # Log configuration to W&B
    wandb_logger.experiment.config.update(config)
    print('after logs to wand')
    # Common variables

    print('before train_selection')
    load_train_selection = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/Selectionsneutrinos_train_from_logit_selection.csv"
    load_val_selection = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/Selectionsneutrinos_val_from_logit_selection.csv"
    load_test_selection = "/groups/icecube/petersen/GraphNetDatabaseRepository/osc_next_database_Peter_and_Morten/Selectionsneutrinos_test_from_logit_selection.csv"
    
    train_selection = pd.read_csv(load_train_selection).reset_index(drop = True)['event_no'].ravel().tolist()
    val_selection = pd.read_csv(load_val_selection).reset_index(drop = True)['event_no'].ravel().tolist()
    test_selection = pd.read_csv(load_test_selection).reset_index(drop = True)['event_no'].ravel().tolist()
    
    
    # load_train_selection = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/using_MP_lvl3/datasets/even_track_cascade_train_event_no.csv"
    # load_val_selection = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/using_MP_lvl3/datasets/even_track_cascade_validation_event_no.csv"
    # load_test_selection = "/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/using_MP_lvl3/datasets/even_track_cascade_test_event_no.csv"
    
    # train_selection = pd.read_csv(load_train_selection).reset_index(drop = True)['0'].ravel().tolist()
    # val_selection = pd.read_csv(load_val_selection).reset_index(drop = True)['0'].ravel().tolist()
    # test_selection = pd.read_csv(load_test_selection).reset_index(drop = True)['0'].ravel().tolist()

    import random
    random.shuffle(train_selection)
    random.shuffle(val_selection)
    random.shuffle(test_selection)

    train_selection = train_selection[:1000000]
    val_selection = val_selection[:100000]
    test_selection = test_selection[:500000]
    
    #event_no_selection = pd.read_csv(load_selection_path)['Unnamed: 0'].values.ravel().tolist()
    #event_no_selection = list(range(0,200000))
    #print(np.min(event_no_selection))
    #print(np.max(event_no_selection))

    #import random
    #random.shuffle(event_no_selection)

    print(f'There are this many train events: {len(train_selection)}')
    print(f'There are this many val events: {len(val_selection)}')
    print(f'There are this many test events: {len(test_selection)}')

    #train_selection, validate_selection, test_selection = split_selection(event_no_selection)

    training_dataloader = make_dataloader(db = config['db'],
                                           selection = train_selection,
                                           pulsemaps = config['pulsemap'],
                                           features = features,
                                           truth = truth,
                                           batch_size = config['batch_size'],
                                           num_workers = config['num_workers'],
                                           shuffle = True)

    validation_dataloader = make_dataloader(db = config['db'],
                                           selection = val_selection,
                                           pulsemaps = config['pulsemap'],
                                           features = features,
                                           truth = truth,
                                           batch_size = config['batch_size'],
                                           num_workers = config['num_workers'],
                                           shuffle = False)

    test_dataloader = make_dataloader(db = config['db'],
                                           selection = test_selection,
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
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    task = BinaryClassificationTask(
        #nb_classes=3,
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=BinaryCrossEntropyLoss(),
        #transform_inference=lambda x: softmax(x,dim=-1),
    )
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-04, "eps": 1e-03},
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
        [config["target"] + "_pred"],
        additional_attributes=[config["target"], "event_no"],
    )

    save_results(config["db"], run_name+'_validation', results, archive, model)

 # Saving predictions to file
    results = get_predictions(
        trainer,
        model,
        test_dataloader,
        [config["target"] + "_pred"],
        additional_attributes=[config["target"], "event_no"],
    )

    save_results(config["db"], run_name+'_test', results, archive, model)

# Main function call
if __name__ == "__main__":
    print('Before main loop')
    main()
