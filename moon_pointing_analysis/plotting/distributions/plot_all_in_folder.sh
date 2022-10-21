# this bash script is meanth to plot everything in the folder
# to recreate the results, follow the steps below.
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")")
# (1) designate directory containing database.
database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/real_data/data_with_reco/moonL4_segspline_exp13_01_merged_with_time_and_reco_and_new_pulsemap.db
# (2) designate output directory for created script.
output_directory=${bash_directory}/test_plot/
# (3) designate the pulsemap used
pulsemap=TWSRTHVInIcePulses

# if directories does not exist for reporting, create them.
mkdir -p ${output_directory};

nohup python ${bash_directory}/distribution_from_sqlite.py -db \
${database_directory} \
-o ${output_directory} \
-p ${pulsemap} \
> log1.out

nohup python ${bash_directory}/pulse_count_and_duration_distribution_from_sqlite.py \
-db ${database_directory} \
-o ${output_directory} \
-p ${pulsemap} > log2.out

echo "Plotting in progress, when done the plots will be available in ${output_directory}."