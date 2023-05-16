#!/bin/bash
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")") # !do not alter!

## Description: This bash script is meant to plot everything in the folder

# to recreate the results, follow the steps below.
# (1) designate directory containing database; data described in folder.
database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/11069/Level2_nugen_numu_IC86.2012.011069_InIcePulses_InIceDSTPulses_SplitInIcePulses_SplitInIceDSTPulses_SRTInIcePulses.db
# (2) to run this shell script; copy file path and execute "bash <file_path>"

sigma=2
cutoff=0.7
log=5000





## do not alter beyond this point ##
#output directory for created plots.
output_directory=${bash_directory}/test_plot/
# save the report file to 
report_name=plot # overwrites old, thus only keeps the one with error
report_directory=/groups/icecube/${USER}/storage/nohup_reports/
# if directories does not exist for reporting, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# save the report file to 
report_location=${report_directory}${report_name}.out

echo "plotting histogram of all features"
nohup python ${bash_directory}/all_features_1Dhistogram.py \
-db ${database_directory} \
-o ${output_directory} \
-s ${sigma} \
-c ${cutoff} \
-l ${log} \
> ${report_location} \
2> ${report_directory}${report_name}.err &