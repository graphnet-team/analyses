#!/bin/bash
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")") # !do not alter!

# Warning!: on hep04; Run gpustat or similar before executing this script and make sure no process is running on the selected gpu.
# Description: training the reconstruction of individual azimuth and zenith direction based on a sqlite3 database.
#              This bash script only works with gpu;

database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/Leon_MC_data/last_one_lvl3MC.db
output_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models_test/

pulsemap=SplitInIcePulses

gpu=1 # the gpu to use [0,1]
batch_size=1024
epochs=100
workers=20
event_numbers=5000 # if number goes over available events; use all events
run_name=LeonMC_data

# save the report file to 
report_name=${run_name} # overwrites old, thus only keeps the one with error
report_directory=/groups/icecube/${USER}/storage/nohup_reports/

# if directories does not exist, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# save the report file to 
report_location=${report_directory}${report_name}.out

python ${bash_directory}/train_multiclass_classification_model.py \
-db ${database_directory} \
-o ${output_directory} \
-n ${event_numbers} \
-p ${pulsemap} \
-g ${gpu} \
-b ${batch_size} \
-e ${epochs} \
-w ${workers} \
-r ${run_name} \
> ${report_location} \
2> ${report_directory}${report_name}.err &
