"""
Converting I3-files to sqlite3 database format.
This is a copy of the example script in graphnet, modified to work in a pipeline, i.e. the script should be kept static.
"""

import logging
import os

from graphnet.utilities.logging import get_logger

from graphnet.data.extractors import (
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
    I3FeatureExtractor,
)
from graphnet.data.parquet import ParquetDataConverter
from graphnet.data.sqlite import SQLiteDataConverter

import argparse

logger = get_logger(level=logging.INFO)

CONVERTER_CLASS = {
    "sqlite": SQLiteDataConverter,
    "parquet": ParquetDataConverter,
}

parser = argparse.ArgumentParser(
    description="processing i3 files to sqlite3 databases"
)
parser.add_argument(
    "--db",
    dest="path_to_db",
    type=str,
    help="path to database [str]",
    default="/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/11069/I3files",
)
parser.add_argument(
    "--gcd",
    dest="gcd_rescue",
    default=None,
    help="define the gcd path, default is 'None'; if set to 'None' it will attempt to find gcd within the file",
)
parser.add_argument(
    "-o", "--out", dest="out", type=str, help="define the output path [str]"
)
parser.add_argument(
    "-k",
    "--keys",
    dest="keys",
    nargs="+",
    help="<Required> list of keys",
    required=True,
)

args = parser.parse_args()


def main_icecube86(backend: str):
    """Convert IceCube-86 I3 files to intermediate `backend` format."""
    # Check(s)

    assert backend in CONVERTER_CLASS

    inputs = [args.path_to_db]
    outdir = args.out

    converter = CONVERTER_CLASS[backend](
        [
            I3FeatureExtractor(keys=args.keys),
            I3TruthExtractor(),
        ],
        outdir,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files(os.path.join(outdir, "merged"))


def main_icecube_upgrade(backend: str):
    """Convert IceCube-Upgrade I3 files to intermediate `backend` format."""
    # Check(s)
    assert backend in CONVERTER_CLASS

    inputs = ["test_data_upgrade_2"]
    outdir = "./temp/test_upgrade"
    workers = 1

    converter = CONVERTER_CLASS[backend](
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCubeUpgrade(
                "I3RecoPulseSeriesMapRFCleaned_mDOM"
            ),
            I3FeatureExtractorIceCubeUpgrade(
                "I3RecoPulseSeriesMapRFCleaned_DEgg"
            ),
        ],
        outdir,
        workers=workers,
        # nb_files_to_batch=10,
        # sequential_batch_pattern="temp_{:03d}",
        # input_file_batch_pattern="[A-Z]{1}_[0-9]{5}*.i3.zst",
        icetray_verbose=1,
    )
    converter(inputs)
    if backend == "sqlite":
        converter.merge_files(os.path.join(outdir, "merged"))


if __name__ == "__main__":
    # backend = "parquet"
    backend = "sqlite"
    main_icecube86(backend)
    # main_icecube_upgrade(backend)
