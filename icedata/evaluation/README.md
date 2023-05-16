# A "framework" to run and manage the results of different GraphNeT components

## Description

This directory contains an implementation of a "framework", which allows to manage multiple versions of GraphNeT components (like `Dataset` or `Detector`) by implementing them in seperate subdirectories, as well as their possibly changing "variables" like number of features, training task, optimizer, learning rate scheduler, etc.
Besides managing allowing training with multiple versions of components, it allows for managing the training results, that is the trained model parameters and the performance metrics (like resolution or roc curve).
The metrics of different versions can then be combined and plotted into a single plot for easy comparison.

## Usage

See help via `python3 . --help`:

```
usage: [-h] [-f FUNCTIONS [FUNCTIONS ...]] [-t TARGETS [TARGETS ...]] [-g GPUS [GPUS ...]] -n RUN_NAMES [RUN_NAMES ...] [-e EPOCHS] [-p PATIENCE]

A script to train, test, generate metrics and run pipelines for variations of graphnet.

optional arguments:
  -h, --help            show this help message and exit
  -f FUNCTIONS [FUNCTIONS ...]
                        what functions to run on targets
  -t TARGETS [TARGETS ...]
                        what targets to run functions on
  -g GPUS [GPUS ...]    what targets to run functions on
  -n RUN_NAMES [RUN_NAMES ...]
                        what targets to run functions on
  -e EPOCHS             what max_epochs to train with
  -p PATIENCE           what patience to use for early stopping during training
```

### Examples

`python3 . -n main-8 idon_tilt-8 -f train_test metrics plot_metrics -t track energy zenith -e 50 -p 5`

`python3 . -n main-8 idon_tilt-8 -f plot_metrics_combined -t track energy zenith`
