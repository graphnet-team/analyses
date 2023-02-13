import os
import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.models.task.reconstruction import ZenithReconstructionWithKappa, AzimuthReconstructionWithKappa, PassOutput1
import torch
from torch.optim.adam import Adam
from graphnet.training.loss_functions import  BinaryCrossEntropyLoss, VonMisesFisher2DLoss, LogCoshLoss
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeUpgrade
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.utils import get_predictions, save_results, make_dataloader
import dill
from graphnet.utilities.logging import get_logger
import numpy as np
from graphnet.models.graph_builders import KNNGraphBuilder

logger = get_logger()

def scale_XYZ(x):
    x[:,0] = x[:,0]/764.431509
    x[:,1] = x[:,1]/785.041607
    x[:,2] = x[:,2]/1083.249944
    return x

def unscale_XYZ(x):
    x[:,0] = 764.431509*x[:,0]
    x[:,1] = 785.041607*x[:,1]
    x[:,2] = 1083.249944*x[:,2]
    return x

def remove_log10(x):
    return torch.pow(10, x)

def transform_to_log10(x):
    return torch.log10(x)


def split_selection(selection):
    """produces a 60%, 20%, 20% split for training, validation and test sets.

    Args:
        selection (pandas.DataFrame): A dataframe containing your selection

    Returns:
        train: indices for training. numpy.ndarray
        validate: indices for validation. numpy.ndarray
        test: indices for testing. numpy.ndarray
    """
    train, validate, test = np.split(selection, [int(.6*len(selection)), int(.8*len(selection))])
    return train.tolist(), validate.tolist(), test.tolist()

def make_dataloaders(selection,
                    data_path,
                    pulsemaps,
                    features,
                    truth,
                    batch_size = 256,
                    num_workers = 60,
                    parquet = False,
                    persistent_workers=True,
                    node_truth=None,
                    node_truth_table=None,
                    string_selection=None,
                    truth_table = 'truth',
                    pid_column = 'pid',
                    interaction_type_column = 'interaction_type',
                    index_column = 'event_no',
                    ):
    train_selection, validate_selection, test_selection = split_selection(selection)
    common_kwargs = dict(
        db=data_path,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        node_truth=node_truth,
        node_truth_table=node_truth_table,
        string_selection=string_selection,

    )

    training_dataloader = make_dataloader(
        selection=train_selection,
        shuffle = True,
        **common_kwargs,
    )

    validation_dataloader = make_dataloader(
        shuffle=False,
        selection=validate_selection,
        **common_kwargs,
    )

    test_dataloader = make_dataloader(
        shuffle=False,
        selection=test_selection,
        **common_kwargs,
    )

    return training_dataloader, validation_dataloader, test_dataloader
                                            
def build_model(run_name, device, archive):
    model = torch.load(os.path.join(archive, f"{run_name}.pth"),pickle_module=dill)
    model.to('cuda:%s'%device[0])
    model.eval()
    model.inference()
    return model 

def train_and_predict_on_validation_set(target,training_dataloader, validation_dataloader, test_dataloader, pulsemap, batch_size, num_workers, n_epochs, device, run_name,archive, patience = 5):
    print(f"features: {features}")
    print(f"truth: {truth}")

    # Building model
    detector = IceCubeUpgrade(graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8))
    if target == 'truth_flag':
        gnn = DynEdge(nb_inputs=detector.nb_outputs,)
    else:
        gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean"],
    )
    if target =='zenith':
        task = ZenithReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=VonMisesFisher2DLoss(),
            loss_weight = None,
        )
        prediction_columns =[target + "_pred", target + "_kappa"]
        additional_attributes=[target, "event_no", "energy"]

    elif target =='azimuth':
        task = AzimuthReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=target,
            loss_function=VonMisesFisher2DLoss(),
            loss_weight = None,
        )
        prediction_columns =[target + "_pred", target + "_kappa"]
        additional_attributes=[target, "event_no", "energy"]

    elif target == 'energy':
        task = PassOutput1(hidden_size=gnn.nb_outputs,
                           target_labels=target, 
                           loss_function=LogCoshLoss(), 
                           transform_target = transform_to_log10, 
                           transform_inference = remove_log10,
                           loss_weight = None,)
        prediction_columns =[target + "_pred"]
        additional_attributes=[target, "event_no"]

    elif target in ['track', 'neutrino']:
        task = BinaryClassificationTask(hidden_size=gnn.nb_outputs,
                                        target_labels=target,
                                        loss_function=BinaryCrossEntropyLoss(),
                                        loss_weight = None,)
        prediction_columns = [target + "_pred"]
        additional_attributes=[target, "event_no", "energy"]
    elif target == 'truth_flag':
        task = BinaryClassificationTask(hidden_size=gnn.nb_outputs,
                                        target_labels=target,
                                        loss_function=BinaryCrossEntropyLoss(), 
                                        loss_weight = None)
        additional_attributes = ['event_no', 'truth_flag', 'dom_type', 'dom_number', 'string']
    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={'lr': 1e-03, 'eps': 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            'milestones': [0, len(training_dataloader) / 2, len(training_dataloader) * n_epochs],
            'factors': [1e-2, 1, 1e-02],
        },
        scheduler_config={
            'interval': 'step',
        },
     )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
        ),
        ProgressBar(),
    ]

    trainer = Trainer(
        default_root_dir=archive + '/' + run_name,
        gpus=device,
        max_epochs=n_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        logger= None
        #resume_from_checkpoint = 
    )
    try:
        trainer.fit(model, training_dataloader, validation_dataloader)
    except KeyboardInterrupt:
        print("[ctrl+c] Exiting gracefully.")
    #model = torch.load('/remote/ceph/user/o/oersoe/upgrade_event_selection/dynedge_pulse_cleaning_truth_flag_SplitInIcePulses.pth', map_location = 'cpu', pickle_module = dill)
    #additional_attributes = ['event_no', 'truth_flag', 'dom_type', 'dom_number', 'string']
    # Saving model
    predict(model,trainer,target,test_dataloader, additional_attributes = additional_attributes, device = device, tag = 'test', prediction_columns = prediction_columns)
    predict(model,trainer,target,validation_dataloader, additional_attributes = additional_attributes, device = device, tag = 'valid', prediction_columns = prediction_columns)
    model.save(os.path.join(archive, f"{run_name}.pth"))
    model.save_state_dict(os.path.join(archive, f"{run_name}_state_dict.pth"))

def get_output_column_names(target):
    if target in ['azimuth', 'zenith']:
        output_column_names=[target + '_pred', target + '_kappa'] 
    if target  in ['track', 'neutrino', 'energy']:
        output_column_names=[target + '_pred']
    if target == 'XYZ':
        output_column_names=['position_x_pred','position_y_pred','position_z_pred']
    if target == 'truth_flag':
        output_column_names = ['truth_flag_pred']

    return output_column_names

def predict(model,trainer,target,dataloader, additional_attributes,device, tag, prediction_columns):
    try:
        del truth[truth.index('interaction_time')]
    except ValueError:
        # not found in list
        pass
    
    device = 'cuda:%s'%device[0]
    model.to(device)
    model.eval()
    model.inference()
    results = get_predictions(
        trainer = trainer,
        model = model,
        dataloader = dataloader,
        prediction_columns = prediction_columns,
        additional_attributes= additional_attributes,
        node_level = True if target == 'truth_flag' else False
    )
    
    save_results(data_path, run_name + '_' + tag, results, archive,model)
    return

# Main function call
if __name__ == "__main__":
    # Run management
    targets = ['track', 'energy','zenith','neutrino'] #, 'zenith', 'energy', 'truth_flag']
    archive = "/remote/ceph/user/o/oersoe/upgrade_event_selection"
    for target in targets:
        weight_column_name = None 
        weight_table_name =  None
        batch_size = 848*2
        data_path = '/mnt/scratch/rasmus_orsoe/databases/upgrade/dev_step4_upgrade_028_with_noise_dynedge_pulsemap_v3_merger_aftercrash/data/dev_step4_upgrade_028_with_noise_dynedge_pulsemap_v3_merger_aftercrash.db'#'/mnt/scratch/rasmus_orsoe/databases//dev_prometheus_full_conversion_first_batch_v2/data/dev_prometheus_full_conversion_first_batch_v2.db'
        device = [0]
        n_epochs = 30
        num_workers = 30
        patience = 5
        if target == 'truth_flag':
            pulsemap =  'SplitInIcePulses' # SRTInIcePulses # 'SplitInIcePulses' #
            node_truth_table = 'SplitInIcePulses_TruthFlags'
            node_truth = ['truth_flag']
        else:
            pulsemap = 'SplitInIcePulses_dynedge_v2_Pulses'
            node_truth_table = None
            node_truth = None
        truth_table = 'truth'
        index_column = 'event_no'

        

        # Configurations
        torch.multiprocessing.set_sharing_strategy('file_system')

        # Constants
        features = FEATURES.UPGRADE
        truth = TRUTH.DEEPCORE[:-1]
        # Common variables
        if target == 'neutrino':
            selection = pd.read_csv('/home/iwsatlas1/oersoe/phd/upgrade_event_selection/training/selections/neutrino_classification.csv').sample(frac = 1)['event_no'].ravel().tolist()#[0:500000]  #selection[0:10000000]
        elif target in ['zenith', 'azimuth', 'energy']:
            selection = pd.read_csv('/home/iwsatlas1/oersoe/phd/upgrade_event_selection/training/selections/regression_selection.csv').sample(frac = 1)['event_no'].ravel().tolist()
        elif target == 'track':
            selection = pd.read_csv('/home/iwsatlas1/oersoe/phd/upgrade_event_selection/training/selections/track_classification.csv').sample(frac = 1)['event_no'].ravel().tolist()
        else:
            assert 1 == 2, f"{target} has no selection."
        # Setup dataloaders
        training_dataloader, validation_dataloader, test_dataloader = make_dataloaders(selection,
                                                                                        data_path = data_path,
                                                                                        pulsemaps = pulsemap,
                                                                                        features = features,
                                                                                        truth = truth,
                                                                                        batch_size = batch_size,
                                                                                        num_workers = num_workers,
                                                                                        parquet = False,
                                                                                        persistent_workers=True,
                                                                                        node_truth=node_truth,
                                                                                        node_truth_table=node_truth_table,
                                                                                        string_selection=None,
                                                                                        truth_table = truth_table,
                                                                                        )

        run_name = f"dynedge_upgrade_reco_{target}_{pulsemap}_{n_epochs}e_p{patience}"
        
        train_and_predict_on_validation_set(target,training_dataloader, validation_dataloader, test_dataloader, pulsemap, batch_size, num_workers, n_epochs, device, run_name,archive)

