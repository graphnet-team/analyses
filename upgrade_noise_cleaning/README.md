

#  Upgrade Pulse Noise Cleaning & Reconstructions with GraphNeT:DynEdge 

![GraphNeT](https://github.com/graphnet-team/graphnet/blob/main/assets/identity/graphnet-logo-and-wordmark.png)
This repository contains documentation for the `SplitInIcePulses_dynedge_v2_Pulses` pulsemap for IceCube Upgrade and associated reconstructions. In addition, all code required to re-train or deploy these models to new i3 files is included. 

The pulsemaps and reconstructions are from DynEdge, a graph neural network implemented in GraphNeT. A technical paper with details on the model is available here.

## Installation of Dependencies

 In order to run the models, you must have GraphNeT installed. Follow the [installation instructions](https://github.com/graphnet-team/graphnet#gear--install) with **IceTray** installation. 

The GPU installation is only required if you want run/retrain the model on GPU. It's not needed for inference on i3 files.

Should future compatibility issues arise, revert to commit `464fa86`(PR [#409](https://github.com/graphnet-team/graphnet/pull/409))
  

##  Applying models to new i3 files

  When packed in a [GraphNeTI3Module](https://github.com/graphnet-team/graphnet/blob/e6110080aff504d0e3903d9cee208f28dc09c428/src/graphnet/deployment/i3modules/graphnet_module.py#L29) and inheriting classes, GraphNeT models are fully compatible with regular IceTray read/write chains. 

**On Cobalt**
If you want to apply these models to new i3 files on cobalt, all you need to do is

- Install GraphNeT (see above) in your favorite IceCube Upgrade CVMFS
- Adjust the input & output folders in `/deployment/apply_to_i3_files.py`
- Run the script

Please note that this code was not intended to be run on clusters.

You can find more examples on using GraphNeT with IceTray [here](https://github.com/graphnet-team/graphnet/tree/main/examples/01_icetray).

**Outside Cobalt**
You will have to copy the models to your destination, adjust paths and follow the **On Cobalt** instructions. 

**Upgrade CVMFS**
I used the following upgrade CVMFS:

    eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.1/setup.sh`
    /cvmfs/icecube.opensciencegrid.org/users/upgrade-sim/software/icetray/branches/icecube_upgrade/build__py3-v4.1.1__osgvo-el7/env-shell.sh

## New I3 Frame Keys
I3 files processed with these models contain the following fields:


### Pulsemaps
 * `SplitInIcePulses_dynedge_v2_Pulses`
>The new pulsemap using DynEdge. This pulsemap was produced from a version of DynEdge that was trained on both neutrinos and muons. A pulse passes the cleaning if DynEdge has given it a score of 0.7 or higher.  Contains all OM types.
>>Model: `/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/pulse_cleaning`
- `SplitInIcePulses_dynedge_v2_Predictions`
>The raw predictions associated with the `SplitInIcePulses_dynedge_v2_Pulses` pulsemap.
 * `SplitInIcePulses_dynedge_v2_Pulses_dEggs_Only`
>A pulsemap that contains only the dEggs from `SplitInIcePulses_dynedge_v2_Pulses`.  
 * `SplitInIcePulses_dynedge_v2_Pulses_mDOMs_Only`
>A pulsemap that contains only the mDOMs from `SplitInIcePulses_dynedge_v2_Pulses`.  
 * `SplitInIcePulses_dynedge_v2_Pulses_pDOMs_Only`
>A pulsemap that contains only the pDOMs from `SplitInIcePulses_dynedge_v2_Pulses`.  

**NOTE**
An early version of this pulsemap is available in some i3 files under the name `SplitInIcePulses_dynedge_Pulses`. This version of the pulsemap cleaner was not trained on muon events and therefore have sub-optimal performance for muons. You are discouraged from using this early version. Please use instead `SplitInIcePulses_dynedge_v2_Pulses`.

### Reconstructions
All of the following models were trained on using `SplitInIcePulses_dynedge_v2_Pulses`.
- `graphnet_dynedge_energy_reconstruction_energy_pred`
> The reconstruction of total, true particle energy. This model was trained on neutrino events only. Output in GeV. 
>>Model: `/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/energy_reconstruction`

___
- `graphnet_dynedge_zenith_reconstruction_zenith_pred`
> The reconstruction of particle zenith. This model was trained on neutrino events only. Output in radians.
>>Model: `/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/zenith_reconstruction`

- `graphnet_dynedge_zenith_reconstruction_zenith_kappa`
>The estimated uncertainty associated with the `graphnet_dynedge_zenith_reconstruction_zenith_pred`. This is the GNN's own estimate of the quality of the reconstruction. Kappa is the concentration parameter of the VonMisesFisher Distribution and is analogues to $\sigma = \frac{1}{\sqrt{kappa}}$ from a normal distribution. This parameter is strongly correlated with recontruction quality and provide nice pull plots if transformed into sigma. For more details, see the GNN paper.
___
- `graphnet_dynedge_track_reconstruction_track_pred`
> The classification score of the track/cascade classifier. This model was trained on a training set where the total sample was subsampled such that there was an equal number of track ($\nu_{\mu,CC}$) and cascade (any other neutrino) examples. The labels were constructed such that 1 = track, 0 = cascade.
>>Model: `/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/track_classification`
`
___
- `graphnet_dynedge_neutrino_reconstruction_neutrino_pred`
> The classification score of the neutrino/muon classifier. This model was trained on a training set where the total sample was subsampled such that there was an equal number of muon and neutrino examples. The labels were constructed such that 1 = neutrino, 0 = muon.
>>Model: `/data/ana/graphnet/upgrade/trained_models/ICRC2023_upgrade_event_selection/neutrino_classification`

