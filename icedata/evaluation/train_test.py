import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from torch.optim.adam import Adam
import wandb

from graphnet.models.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.models import Model
from graphnet.models.detector.icecube import IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder


import common


def train_test(args: common.Args, vals: common.Vals):
    print()
    print(f'train({args.run_name=}, {args.target=})')

    # Make output dir
    os.makedirs(args.archive.root_str, exist_ok=True)

    # Setup Training
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
        ),
        ProgressBar(),
    ]

    wandb_logger = WandbLogger(
        project="graphnet",
        entity="timephy",
        save_dir="./wandb/",
        log_model=False,
        name=f'{args.run_name}:{args.target}',
    )
    wandb_logger.experiment.config.update(args.__dict__)
    wandb_logger.experiment.config.update({
        'features': vals.detector.features,
        'optimizer_kwargs': vals.model._optimizer_kwargs,
        'scheduler_kwargs': vals.model._scheduler_kwargs,
        'vals_class': vals.__class__.__name__,
    }, allow_val_change=True)

    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=wandb_logger,
    )

    # Training
    trainer.fit(vals.model, vals.training_dataloader, vals.validation_dataloader)

    # Predicting
    results = common.get_predictions(
        target=args.target,
        model=vals.model,
        trainer=trainer,
        test_dataloader=vals.test_dataloader,
    )

    vals.model.save_state_dict(args.archive.state_dict_str)
    vals.model.save(args.archive.model_str)
    results.to_csv(args.archive.results_str)

    wandb.finish()  # 'reset' to 'split' multiple runs
