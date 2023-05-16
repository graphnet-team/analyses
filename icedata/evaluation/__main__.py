from pipeline import run_pipeline
from metrics import generate_metrics, plot_metrics, plot_metrics_combined
from train_test import train_test
import idon_transformer_old
import rde
import idon_tilt
import main as _main
import common
import argparse
import torch
from graphnet.data.constants import FEATURES, TRUTH
from typing import List, Optional, Union
from pathlib import Path


# Configurations
torch.multiprocessing.set_sharing_strategy('file_system')

# Constants
_features = FEATURES.DEEPCORE
_truth = TRUTH.DEEPCORE

try:
    del _truth[_truth.index('interaction_time')]
except ValueError:
    # not found in list
    pass


def convert_model(args: common.Args, vals: common.Vals):
    print()
    print(f'===== convert_model({args.target=}) =====')

    # Building model
    vals.model.load_state_dict(args.archive.state_dict_str)
    vals.model.save(args.archive.model_str)


def main():
    # Config
    functions_all = ['convert_model', 'train_test', 'metrics', 'plot_metrics', 'pipeline', 'plot_metrics_combined']
    targets_all: list[common.Target] = ['track', 'energy', 'zenith']
    run_names_all = [
        'main-10',
        'main-8',
        'main-4',
        #
        'idon-8',
        'idon-6',
        #
        'idon_tilt-8',
        'idon_tilt_transformed-8',
        #
        'idon_tilt_old-8',
        'idon_transformer_old-8',
        'idon_old-8',
        #
        'rde-8',
        'rde_transformed-8',
    ]

    # Parser
    parser = argparse.ArgumentParser(
        description='A script to train, test, generate metrics and run pipelines for variations of graphnet.')

    parser.add_argument('-f', dest='functions', nargs='+',
                        default=['train_test', 'metrics', 'plot_metrics'],
                        help='what functions to run on targets')
    parser.add_argument('-t', dest='targets', nargs='+',
                        default=targets_all,
                        help='what targets to run functions on')
    parser.add_argument('-g', dest='gpus', nargs='+', type=int,
                        default=[3],
                        help='what targets to run functions on')
    parser.add_argument('-n', dest='run_names', nargs='+',
                        required=True,
                        help='what targets to run functions on')
    parser.add_argument('-e', dest='epochs', type=int,
                        default=50,
                        help='what max_epochs to train with')
    parser.add_argument('-p', dest='patience', type=int,
                        default=5,
                        help='what patience to use for early stopping during training')

    args = parser.parse_args()
    print(f'{args=}')

    targets = args.targets
    functions = args.functions
    run_names = args.run_names
    gpus = args.gpus
    epochs = args.epochs
    patience = args.patience

    archive_base = Path(f'/remote/ceph/user/t/timg/archive')

    # Run
    # args_dict = {}
    # for run_name in run_names_all:
    #     args_dict[run_name] = {}
    #     for target in targets_all:
    #         args_dict[run_name][target] = common.Args(
    #             run_name=run_name,
    #             target=target,

    #             database=Path('/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db'),
    #             pulsemap='SRTTWOfflinePulsesDC',
    #             features=_features,
    #             truth=_truth,

    #             batch_size=512,
    #             num_workers=30,
    #             gpus=gpus,

    #             max_epochs=50,
    #             patience=5,

    #             archive=common.Archive(archive_base.joinpath(f'{run_name}/{target}'))
    #         )
    args_dict = {}
    for run_name in run_names:
        args_dict[run_name] = {}
        for target in targets:
            args_dict[run_name][target] = common.Args(
                run_name=run_name,
                target=target,

                database=Path('/remote/ceph/user/t/timg/dev_lvl7_robustness_muon_neutrino_0000.db'),
                pulsemap='SRTTWOfflinePulsesDC',
                features=_features,
                truth=_truth,

                batch_size=512,
                num_workers=30,
                gpus=gpus,

                max_epochs=epochs,
                patience=patience,

                archive=common.Archive(archive_base.joinpath(f'{run_name}/{target}'))
            )

    # Print overview
    def print_status(
        args_active: Union[common.Args, List[common.Args]] = [],
        function_active: Union[str, List[str]] = []
    ):
        from colorama import Fore, Back, Style

        if not isinstance(args_active, list):
            args_active = [args_active]
        if not isinstance(function_active, list):
            function_active = [function_active]

        def if_(function=None, run_name=None, target=None):
            if function is not None and function not in functions:
                return False
            if run_name is not None and run_name not in run_names:
                return False
            if target is not None and target not in targets:
                return False
            return True

        print()
        print('The selection of targets that functions will be run on:')
        for run_name in run_names:
            for target in targets:
                prefix = Fore.LIGHTRED_EX + '◉' if args_dict[run_name][target] in args_active else \
                    Fore.GREEN + '◉' if if_(run_name=run_name, target=target) else \
                    Fore.LIGHTBLACK_EX + '◯'
                print(
                    prefix,
                    f'{run_name:8} {target:8} {args_dict[run_name][target]}' +
                    Style.RESET_ALL
                )

        print()
        print('The selection of functions that will be run:')
        for function in functions_all:
            prefix = Fore.LIGHTRED_EX + '◉' if function in function_active else \
                Fore.GREEN + '◉' if if_(function=function) else \
                Fore.LIGHTBLACK_EX + '◯'
            print(
                prefix,
                f'{function}' +
                Style.RESET_ALL
            )

    print_status()

    # Execute
    for run_name in run_names:
        for target in targets:
            if 'convert_model' in functions:
                args = args_dict[run_name][target]
                vals = get_vals(args)
                print_status(args_active=args, function_active='convert_model')
                convert_model(args, vals)

            if 'train_test' in functions:
                args = args_dict[run_name][target]
                vals = get_vals(args)
                print_status(args_active=args, function_active='train_test')
                train_test(args, vals)

            if 'metrics' in functions:
                args = args_dict[run_name][target]
                print_status(args_active=args, function_active='metrics')
                generate_metrics(args)

            if 'plot_metrics' in functions:
                args = args_dict[run_name][target]
                print_status(args_active=args, function_active='plot_metrics')
                plot_metrics(args)

        if 'pipeline' in functions:
            args_list = list(args_dict[run_name].values())
            print_status(args_active=args_list, function_active='pipeline')
            run_pipeline(args_list)

    if 'plot_metrics_combined' in functions:
        for target in targets:
            args_list = [args_dict[run_name][target] for run_name in run_names]
            print_status(args_active=args_list, function_active='plot_metrics_combined')
            plot_metrics_combined(args_list, path_out=str(archive_base.joinpath(f'metrics_{target}.png').absolute()))
            plot_metrics_combined(args_list, path_out=str(archive_base.joinpath(f'metrics_{target}.svg').absolute()))


def get_vals(args: common.Args) -> common.Vals:
    # MAIN
    if args.run_name.startswith('main-10'):
        return _main.Vals_MAIN(args, nb_nearest_neighbours=10)

    if args.run_name.startswith('main-8'):
        return _main.Vals_MAIN(args, nb_nearest_neighbours=8)

    elif args.run_name.startswith('main-4'):
        return _main.Vals_MAIN(args, nb_nearest_neighbours=4)

    elif args.run_name.startswith('main-3'):
        return _main.Vals_MAIN(args, nb_nearest_neighbours=3)

    # OLD
    elif args.run_name.startswith('idon_transformer_old-8'):
        return idon_transformer_old.Vals_IDON(args, nb_nearest_neighbours=8)

    # IDON_Tilt
    elif args.run_name.startswith('idon_tilt-8'):
        return idon_tilt.Vals_IDON_Tilt(args, nb_nearest_neighbours=8)
    elif args.run_name.startswith('idon_tilt_transformed-8'):
        return idon_tilt.Vals_IDON_Tilt_Transformed(args, nb_nearest_neighbours=8)

    # RDE
    elif args.run_name.startswith('rde-8'):
        return rde.Vals_RDE(args, nb_nearest_neighbours=8)
    elif args.run_name.startswith('rde_transformed-8'):
        return rde.Vals_RDE_Transformed(args, nb_nearest_neighbours=8)

    # else
    else:
        raise Exception('run_name not found')


if __name__ == '__main__':
    print('main()')
    main()
