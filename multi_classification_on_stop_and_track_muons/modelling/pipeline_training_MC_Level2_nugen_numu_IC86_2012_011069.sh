#!/bin/bash
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")") # !do not alter!

# Warning!: on hep04; Run gpustat or similar before executing this script and make sure no process is running on the selected gpu.
# Description: training the reconstruction of individual azimuth and zenith direction based on a sqlite3 database.
#              This bash script only works with gpu;

database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/11069/Level2_nugen_numu_IC86.2012.011069_InIcePulses_InIceDSTPulses_SplitInIcePulses_SplitInIceDSTPulses_SRTInIcePulses.db
output_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/trained_models/Level2_nugen_numu_IC86_2012_011069/

pulsemap=SplitInIcePulses

gpu=0 # the gpu to use [0,1]
batch_size=128
epochs=100
workers=10
event_numbers=50 # if number goes over available events; use all events
run_name=Level2_nugen_numu_IC86_2012_11069

# save the report file to 
report_name=${run_name} # overwrites old, thus only keeps the one with error
report_directory=/groups/icecube/${USER}/storage/nohup_reports/

# if directories does not exist, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# save the report file to 
report_location=${report_directory}${report_name}.out

python ${bash_directory}/train_reconstruction_individual_azimuth_zenith.py \
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
