from typing import List


import common

from graphnet.data.pipeline import InSQLitePipeline
from idon_tilt.dataset import SQLiteDataset_IDON_Tilt


def run_pipeline(args_list: List[common.Args]):
    assert len(args_list) > 0

    def get_output_column_names(target: common.Target):
        if target in ['azimuth', 'zenith']:
            return [target + '_pred', target + '_kappa']
        elif target in ['track', 'neutrino', 'energy']:
            return [target + '_pred']

    def build_module_dictionary(args_list: List[common.Args]):
        module_dict = {}
        for args in args_list:
            module_dict[args.target] = {}
            module_dict[args.target]['path'] = args.archive.model_str
            module_dict[args.target]['output_column_names'] = get_output_column_names(args.target)
        return module_dict

    # Build Pipeline
    pipeline = InSQLitePipeline(
        module_dict=build_module_dictionary(args_list),
        features=args_list[0].features,
        truth=args_list[0].truth,
        device=f'cuda:{args_list[0].gpus[0]}',
        batch_size=args_list[0].batch_size,
        n_workers=args_list[0].num_workers,
        pipeline_name='pipeline',
        outdir=str(args_list[0].archive.root.joinpath('../pipeline')),
        dataset_class=SQLiteDataset_IDON_Tilt,  # NOT THIS IN CASE OF SWITCHING!!!
    )

    # Run Pipeline
    pipeline(args_list[0].database_str, args_list[0].pulsemap)
