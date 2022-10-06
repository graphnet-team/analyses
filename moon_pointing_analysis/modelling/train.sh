#!/bin/bash

db="/groups/icecube/asogaard/data/sqlite/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db"
out="/groups/icecube/peter/workspace/graphnetmoon/graphnet/studies/Moon_Pointing_Analysis/modelling/TrainedModels/TestData/"
#gpu=1

python train_model_Azimuthal.py --database ${db} --output ${out} #--gpu ${gpu}
