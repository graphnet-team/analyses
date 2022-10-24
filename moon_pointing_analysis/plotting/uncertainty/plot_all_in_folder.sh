#!/usr/bin/env python3
# !do not alter!
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")")

## Description: This bash script is meant to plot everything in the folder

# to recreate the results, follow the steps below.
# (1) designate directory containing database; data described in folder.
database_directory=(\
/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/trained_models/dynedge_azimuth_example/results.csv \
/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/trained_models/dynedge_zenith_example/results.csv \
)
# (2) designate the pulsemap used.
pulsemap=TWSRTHVInIcePulses
# (3) specifically for the heatmap, define the number of bins.
bins=500
# (4) to run this shell script; copy file path and execute "bash <file_path>"








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

nohup python ${bash_directory}/uncertainty_test.py \
--database ${database_directory[@]} \
--bins ${bins} \
-o ${output_directory} \
-p ${pulsemap} \
> ${report_location}

#nohup python ${bash_directory}/reconstruction.py \
#-db ${database_directory[@]} \
#-o ${output_directory} \
#-p ${pulsemap} \
#> ${report_location}