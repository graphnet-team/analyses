#!/bin/bash
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")") # !do not alter!

# Warning!: on hep04; Run gpustat or similar before executing this script and make sure no process is running on the selected gpu.
# Description: training the reconstruction of individual azimuth and zenith direction based on a sqlite3 database.
#              This bash script only works with gpu;

database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Datasets/last_one_lvl3MC.db
output_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/trained_models/

pulsemap=SRTInIcePulses


accelerator="cpu" # "cpu" or "gpu"; if set to cpu, gpu_number has no effect.
gpu_number=1 # the gpu to use [0,1];
batch_size=1024
epochs=100
patience=10
workers=10
event_numbers=5000 #22000000 # if number goes over available events; use all events
run_name=LeonMC_data_5k #22m

class_options="{1:0,-1:0,13:1,-13:1,12:2,-12:2,14:2,-14:2,16:2,-16:2}"
target='pid'

# save the report file to 
report_name=${run_name} # overwrites old, thus only keeps the one with error
report_directory=/groups/icecube/${USER}/storage/nohup_reports/

# if directories does not exist, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory}; 

# save the report file to 
report_location=${report_directory}${report_name}.out
echo "report location: " ${report_location} ".out"
echo "error report location: " ${report_directory}${report_name}".err"

python ${bash_directory}/train_classification_model.py \
-db ${database_directory} \
-o ${output_directory} \
-c ${class_options} \
-t ${target} \
-n ${event_numbers} \
-p ${pulsemap} \
-a ${accelerator} \
-g ${gpu_number} \
-b ${batch_size} \
-e ${epochs} \
-s ${patience} \
-w ${workers} \
-r ${run_name} \
> ${report_location} \
2> ${report_directory}${report_name}.err &
