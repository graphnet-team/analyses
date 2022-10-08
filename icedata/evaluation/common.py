from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple
from pandas import DataFrame
from pytorch_lightning import Trainer

import torch
from torch.utils.data import DataLoader
from graphnet.data.sqlite_dataset import SQLiteDataset

from graphnet.data.utils import get_desired_event_numbers, get_even_track_cascade_indicies
from graphnet.models.detector.detector import Detector
from graphnet.models.gnn.gnn import GNN
from graphnet.models.training.utils import get_predictions as _get_predictions, make_train_validation_dataloader, make_dataloader
from graphnet.components.loss_functions import BinaryCrossEntropyLoss, LogCoshLoss, VonMisesFisher2DLoss
from graphnet.models.task.task import Task
from graphnet.models.model import Model
from graphnet.models.task.reconstruction import PassOutput1, BinaryClassificationTask, EnergyReconstruction, ZenithReconstruction, ZenithReconstructionWithKappa

from sklearn.model_selection import train_test_split

from torch.nn.modules.loss import L1Loss, MSELoss

from weighted_loss import LogCoshLoss_Weighted

Target = Literal["track", "energy", "zenith"]


@dataclass
class Archive:
    '''Utility to create all paths used.'''
    root: Path

    @property
    def root_str(self):
        return str(self.root.absolute())

    @property
    def state_dict_str(self):
        return str(self.root.joinpath('state_dict.pth').absolute())

    @property
    def model_str(self):
        return str(self.root.joinpath('model.pth').absolute())

    @property
    def results_str(self):
        return str(self.root.joinpath('results.csv').absolute())

    @property
    def roc_auc_plot_str(self):
        return str(self.root.joinpath('roc_auc.png').absolute())

    @property
    def roc_csv_str(self):
        return str(self.root.joinpath('roc.csv').absolute())

    @property
    def auc_file_str(self):
        return str(self.root.joinpath('auc.npy').absolute())

    @property
    def resolution_plot_str(self):
        return str(self.root.joinpath('resolution.png').absolute())

    @property
    def resolution_csv_str(self):
        return str(self.root.joinpath('resolution.csv').absolute())


@dataclass
class Args:
    '''Carries all arguments used in individual runs.'''
    run_name: str  # nnb-8, nnb-4, ...
    target: Target  # track, energy, zenith

    database: Path
    pulsemap: str
    features: List[str]
    truth: List[str]

    batch_size: int
    num_workers: int
    gpus: List[int]

    max_epochs: int
    patience: int

    archive: Archive

    @property
    def database_str(self):
        return str(self.database.absolute())

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f'(run_name={self.run_name!r}, target={self.target!r})')


@dataclass
class Vals:
    '''Carries all model and data related info created from args'''
    detector: Detector
    gnn: GNN
    task: Task
    model: Model

    training_dataloader: DataLoader
    validation_dataloader: DataLoader
    test_dataloader: DataLoader


def get_predictions(*, target: Target, model: Model, trainer: Trainer, test_dataloader: DataLoader) -> DataFrame:
    '''Get predictions for given target from model using trainer from test_dataloader data.

    Args:
        target (Target): The target.
        model (_type_): The model.
        trainer (Trainer): The trainer.
        test_dataloader (DataLoader): The dataloader.

    Returns:
        DataFrame: Contains prediction, truth, energy, event_no and kappa in case of zenith.
    '''
    if target == 'track':
        return _get_predictions(
            trainer,
            model,
            test_dataloader,
            prediction_columns=[target + '_pred'],
            additional_attributes=[target, 'event_no', 'energy'],
        )

    elif target == 'energy':
        return _get_predictions(
            trainer,
            model,
            test_dataloader,
            prediction_columns=[target + '_pred'],
            additional_attributes=[target, 'event_no'],
        )

    elif target == 'zenith':
        return _get_predictions(
            trainer,
            model,
            test_dataloader,
            prediction_columns=[target + '_pred', target + '_kappa'],
            additional_attributes=[target, 'event_no', 'energy'],
        )


def get_task(target: Target, gnn: GNN) -> Task:
    '''Returns a task for given target and matches gnn size.

    Args:
        target (Target): The target.
        gnn (GNN): The gnn of which the size is used.

    Returns:
        Task: The Task.
    '''
    if target == 'track':
        return BinaryClassificationTask(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=BinaryCrossEntropyLoss(),
        )

    elif target == 'energy':
        # started testing this on 10.07.22
        # return PassOutput1(
        #     hidden_size=gnn.nb_outputs,
        #     target_labels=target,
        #     loss_function=MSELoss(),
        #     transform_target=torch.log10,
        #     transform_inference=lambda x: torch.pow(10, x)
        # )
        # used the following for most of the training
        # return PassOutput1(
        #     hidden_size=gnn.nb_outputs,
        #     target_labels=target,
        #     loss_function=LogCoshLoss(),
        #     transform_target=torch.log10,
        #     transform_inference=lambda x: torch.pow(10, x)
        # )
        return PassOutput1(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=LogCoshLoss_Weighted(),
            transform_target=torch.log10,
            transform_inference=lambda x: torch.pow(10, x)
        )

    elif target == 'zenith':
        return ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=VonMisesFisher2DLoss(),
        )


def get_selections(args: Args):
    if args.target == 'track':
        train_valid_selection, test_selection = get_even_track_cascade_indicies(args.database)

    elif args.target == 'energy' or args.target == 'zenith':
        selection = get_desired_event_numbers(
            args.database,
            10000000000,  # 10000000000
            fraction_muon=0, fraction_nu_e=0.34, fraction_nu_mu=0.33, fraction_nu_tau=0.33  # type: ignore
        )
        train_valid_selection, test_selection = train_test_split(selection, test_size=0.25, random_state=42)

    else:
        raise Exception('target does not match')

    return train_valid_selection, test_selection


def get_dataloaders(args: Args, *, dataset_class=SQLiteDataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    '''Creates three Dataloaders from args.database and selections that match args.target.

    Args:
        args (Args): All arguments.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Training, Validation, Test.
    '''
    train_valid_selection, test_selection = get_selections(args)

    training_dataloader, validation_dataloader = make_train_validation_dataloader(  # type: ignore
        db=args.database_str,
        selection=train_valid_selection,
        pulsemaps=args.pulsemap,
        features=args.features,
        truth=args.truth,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=0.33,
        dataset_class=dataset_class,
    )
    test_dataloader = make_dataloader(
        db=args.database_str,
        pulsemaps=args.pulsemap,
        features=args.features,
        truth=args.truth,
        selection=test_selection,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        dataset_class=dataset_class,
    )
    return training_dataloader, validation_dataloader, test_dataloader
