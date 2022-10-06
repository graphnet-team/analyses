"""Minimum working example (MWE) to use SQLiteDataConverter."""

import logging
import os
import argparse

from graphnet.utilities.logging import get_logger

from graphnet.data.extractors import (
    I3FeatureExtractorIceCube86,
    I3FeatureExtractorIceCubeUpgrade,
    I3RetroExtractor,
    I3TruthExtractor,
)
from graphnet.data.sqlite.sqlite_dataconverter import SQLiteDataConverter

logger = get_logger(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='processing i3 files to sqlite3 databases')
parser.add_argument('--db', dest='path_to_db', type=str, help='path to database [str]')
parser.add_argument('--pulse', dest='pulsemap', type=str, help='path to database [str]')
parser.add_argument('--gcd', dest='gcd_rescue', default=None, help='sum the integers (default: find the max)')
parser.add_argument('--out', dest='out')

args = parser.parse_args()


def main_icecube86():
    """Main script function."""
    paths = [
        args.path_to_db
    ]
    pulsemap = args.pulsemap
    gcd_rescue = args.gcd_rescue
    outdir = "/groups/icecube/qgf305/storage/I3_files/Sebastian_MoonL4/data_out"

    converter = SQLiteDataConverter(
        [
            I3TruthExtractor(),
            I3RetroExtractor(),
            I3FeatureExtractorIceCube86(pulsemap),
        ],
        outdir,
        gcd_rescue,
    )
    converter(paths)
    converter.merge_files()


def main_icecube_upgrade():
    """Main script function."""
    basedir = "/groups/icecube/asogaard/data/IceCubeUpgrade/nu_simulation/detector/step4/"
    paths = [os.path.join(basedir, "step4")]
    gcd_rescue = os.path.join(
        basedir, "gcd/GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2"
    )
    outdir = "/groups/icecube/asogaard/temp/sqlite_test_upgrade"
    workers = 10

    converter = SQLiteDataConverter(
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
        gcd_rescue,
        workers=workers,
        nb_files_to_batch=1000,
        # sequential_batch_pattern="temp_{:03d}",
        input_file_batch_pattern="[A-Z]{1}_[0-9]{5}*.i3.zst",
        verbose=1,
    )
    converter(paths)


if __name__ == "__main__":
    main_icecube86()
    # main_icecube_upgrade()
