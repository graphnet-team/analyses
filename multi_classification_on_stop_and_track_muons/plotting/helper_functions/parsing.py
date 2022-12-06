import argparse

def get_plotArgs():
    parser = argparse.ArgumentParser(
    description="plotting the predicted zenith and azimuth vs truth."
    )
    parser.add_argument(
        "-mc_r",
        "--mc_results",
        dest="MC_csv_path",
        type=str,
        help="<required> path to monte carlo results",
        default="/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Inference/pid_Leon_MC_Full_db.csv",
    )
    parser.add_argument(
        "-rd_r",
        "--rd_results",
        dest="RD_csv_path",
        type=str,
        help="<required> path to inference results",
        default="/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Inference/pid_Leon_RD_results_new_model.csv",
    )
    parser.add_argument(
        "-mc_db",
        "--mc_database",
        dest="MC_db_path",
        type=str,
        help="<required> path to monte carlo raw database",
        default="/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Datasets/last_one_lvl3MC.db",
    )
    parser.add_argument(
        "-rd_db",
        "--rd_database",
        dest="RD_db_path",
        type=str,
        help="<required> path to real data raw database",
        default="/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Datasets/last_one_lvl3.db",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        type=str,
        help="<required> the output path [str]",
        default="/groups/icecube/petersen/GraphNetDatabaseRepository/example_results/debug/",
    )
    return parser.parse_args()