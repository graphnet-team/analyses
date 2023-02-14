from glob import glob
from os.path import join
from typing import List, Dict, Any

from graphnet.deployment.i3modules import I3InferenceModule, I3PulseCleanerModule
from graphnet.deployment.i3modules import GraphNeTI3Deployer
from graphnet.data.extractors.i3featureextractor import I3FeatureExtractorIceCubeUpgrade
from graphnet.data.constants import FEATURES

def construct_modules(model_dict: Dict[str, Any], gcd_file: str) -> Dict[str, Any]:
    features = FEATURES.UPGRADE
    deployment_modules = []
    for model_name in model_dict.keys():
        model_path = model_dict[model_name]['model_path']
        prediction_columns = model_dict[model_name]['prediction_columns']
        pulsemap = model_dict[model_name]['pulsemap']
        extractor = I3FeatureExtractorIceCubeUpgrade(pulsemap = pulsemap)
        deployment_modules.append(I3InferenceModule(pulsemap = pulsemap,
                                                    features = features,
                                                    pulsemap_extractor = extractor,
                                                    model = model_path,
                                                    gcd_file = gcd_file,
                                                    prediction_columns = prediction_columns,
                                                    model_name = model_name))
    return deployment_modules


def deploy_models(input_files: List[str], 
                  output_folder: str, 
                  gcd_file: str, 
                  n_workers: int):
    """ Applies pulse cleaning, energy & zenith reconstruction, 
        track & neutrino classifiers to i3 files. 
        """
    pulsemap = 'SplitInIcePulses_dynedge_v2_Pulses'
    model_dict = {}
    model_dict['graphnet_dynedge_energy_reconstruction'] = {'model_path': '/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/energy_reconstruction/energy_reconstruction_updated.pth',
                                                            'prediction_columns': ['energy_pred'],
                                                            'pulsemap': pulsemap}

    model_dict['graphnet_dynedge_track_classification'] = {'model_path': '/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/track_classification/dynedge_upgrade_reco_track_SplitInIcePulses_dynedge_v2_Pulses_30e_p5_test_model.pth',
                                                            'prediction_columns': 'track_pred',
                                                            'pulsemap': pulsemap}

    model_dict['graphnet_dynedge_neutrino_classification'] = {'model_path': '/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/neutrino_classification/dynedge_upgrade_reco_neutrino_SplitInIcePulses_dynedge_v2_Pulses_30e_p5_test_model.pth',
                                                            'prediction_columns': 'neutrino_pred',
                                                            'pulsemap': pulsemap}

    model_dict['graphnet_dynedge_zenith_reconstruction'] = {'model_path': '/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/zenith_reconstruction/dynedge_upgrade_reco_zenith_SplitInIcePulses_dynedge_v2_Pulses_30e_p5_test_model.pth',
                                                            'prediction_columns': ['zenith_pred', 'zenith_kappa'],
                                                            'pulsemap': pulsemap}

    pulse_cleaner = I3PulseCleanerModule(pulsemap = 'SplitInIcePulses',
                                        features = FEATURES.UPGRADE,
                                        pulsemap_extractor = I3FeatureExtractorIceCubeUpgrade(pulsemap = 'SplitInIcePulses'),
                                        model = '/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/pulse_cleaning/dynedge_pulse_cleaning_truth_flag_SplitInIcePulses_50e_p5_full.pth',
                                        model_name = 'dynedge_v2',
                                        prediction_columns = 'truth_flag_pred',
                                        gcd_file = gcd_file)
    deployment_modules = construct_modules(model_dict = model_dict, gcd_file = gcd_file)

    deploy_this = [pulse_cleaner]

    for module in deployment_modules:
        deploy_this.append(module)

    deployer = GraphNeTI3Deployer(graphnet_modules = deploy_this,
                                n_workers = n_workers,
                                gcd_file = gcd_file,
                                )

    deployer.run(input_files = input_files,
                output_folder = output_folder,)


if __name__ == '__main__':
    n_workers = 2
    input_folders = ['your/input_i3_files']
    input_files = []
    for folder in input_folders:
        input_files.extend(glob(join(folder, "*.i3*")))
    ## reads in all files in "input_folder" and writes those files to "output_folder" with changes seen in "submit_to_frame"

    output_folder = 'your_output_folder'
    gcd_file = '/data/sim/IceCubeUpgrade/geometries/GCDs/GeoCalibDetectorStatus_ICUpgrade.v58.mixed.V0.i3.bz2'
    deploy_models(input_files = input_files, 
                  output_folder = output_folder, 
                  gcd_file = gcd_file,
                  n_workers = n_workers)
