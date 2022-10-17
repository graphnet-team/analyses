#!/usr/bin/env bash
# Notes: Contains a small scale test on hep, containing 3 I3-files

# to recreate the results, follow the steps below.
# (1) designate directory containing I3-files, including gcd file.
database_directory=/data/user/sschindler/zeuthenCluster_moonL4_exp13_01 # note: gcd contained within I3-files
# (2) designate output directory for created script.
output_directory=/data/user/${USER}/storage/moon_pointing_analysis/real_data/moonL4_segspline_exp13_01_redo
# (3) report output location and name
report_directory=/data/user/${USER}/storage/nohup_reports/
report_name=MC
# (4) designate the feature keys to extract; found via investigating the I3-files using dataio-shovel in IceTray.
keys=(TWSRTHVInIcePulses, SplitInIcePulses, HVInIcePulses, InIceDSTPulses, SegmentedSpline, SplineMPE, SplineMPEIC, RNNReco, RNNReco_sigma)


## do not alter beyond this point ##
# date for report name
TIMESTAMP=$(date "+%H%M%S")

# if directories does not exist for reporting, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# save the report file to 
report_location=${report_directory}${report_name}${TIMESTAMP}.out

# run script based on bash script location
nohup python $(dirname -- "$(readlink -f "${BASH_SOURCE}")")/convert_i3_files.py \
--db ${database_directory} \
--out ${output_directory} \
--keys ${keys[@]} \
> ${report_location}