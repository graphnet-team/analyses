#!/bin/bash
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")") # !do not alter!

## Description: This bash script is meant to plot everything in the folder

# to recreate the results, follow the steps below.
# (1) designate directory containing database; data described in folder.
database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/real_data/data_with_reco/moonL4_segspline_exp13_01_merged_with_time_and_reco_and_new_pulsemap.db
# (2) to run this shell script; copy file path and execute "bash <file_path>"








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
> ${report_location} &