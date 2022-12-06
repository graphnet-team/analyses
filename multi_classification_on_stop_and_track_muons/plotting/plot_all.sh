#!/usr/bin/env python3
# !do not alter!
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")")

## Description: This bash script is meant to plot everything in article

# to recreate the results, follow the steps below.
# (1) designate directory containing databases and csv for real data (rd) and monte carlo (mc).
MC_csv_path=/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Inference/pid_Leon_MC_Full_db.csv
RD_csv_path=/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Inference/pid_Leon_RD_results_new_model.csv
MC_database_path=/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Datasets/last_one_lvl3MC.db
RD_database_path=/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/Datasets/last_one_lvl3.db
# (2) designate directory for output; preferably use a folder with model name and end with "/".
output_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/multi_classification_stop_track_muon/plotting/pid_Leon_MC_Full_db/
# (3) to run this shell script; copy file path and execute "bash <file_path>"


### do not alter beyond this point ###
# save the report file to 
report_name=plot # overwrites old, thus only keeps the one with error
report_directory=/groups/icecube/${USER}/storage/nohup_reports/
# if directories does not exist for reporting, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# save the report file to 
report_location=${report_directory}${report_name}.out

## Classification Performance
echo "plotting confusion matrix for MC results"
nohup python ${bash_directory}/Classification_performance/plot_confusion_matrix.py \
-mc_r ${MC_csv_path} \
-o ${output_directory} \
> ${report_location} \
2> ${report_directory}${report_name}.err &


echo "plotting probability heat maps for RD and MC results"
nohup python ${bash_directory}/Classification_performance/plot_probability_heatmaps.py \
-mc_r ${MC_csv_path} \
-rd_r ${RD_csv_path} \
-o ${output_directory} \
> ${report_location} \
2> ${report_directory}${report_name}.err &


echo "plotting receiver operating curve for MC results"
nohup python ${bash_directory}/Classification_performance/plot_roc.py \
-mc_r ${MC_csv_path} \
-o ${output_directory} \
> ${report_location} \
2> ${report_directory}${report_name}.err &


## distributions
echo "plotting prediction for MC results"
nohup python ${bash_directory}/distributions/PID_predictions.py \
-mc_r ${MC_csv_path} \
-o ${output_directory} \
> ${report_location} \
2> ${report_directory}${report_name}.err &


echo "plotting prediction fit for RD and MC results"
nohup python ${bash_directory}/distributions/PID_predictions_fit.py \
-mc_r ${MC_csv_path} \
-rd_r ${RD_csv_path} \
-o ${output_directory} \
> ${report_location} \
2> ${report_directory}${report_name}.err &


echo "plotting logits predictions for RD and MC results"
nohup python ${bash_directory}/distributions/PID_predictions_logit.py \
-mc_r ${MC_csv_path} \
-rd_r ${RD_csv_path} \
-o ${output_directory} \
> ${report_location} \
2> ${report_directory}${report_name}.err &


echo "plotting logit prediction fit for RD and MC results"
nohup python ${bash_directory}/distributions/PID_predictions_logit_fit.py \
-mc_r ${MC_csv_path} \
-rd_r ${RD_csv_path} \
-o ${output_directory} \
> ${report_location} \
2> ${report_directory}${report_name}.err &
