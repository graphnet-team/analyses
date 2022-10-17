#!/usr/bin/env bash
# Notes: points to the entire 1.1TB dataset of monte carlo numu data.

# to recreate the results, follow the steps below.
# (1) designate directory containing I3-files, including gcd file.
database_directory=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/09000-09999/
#gcd=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/08000-08999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz
# (2) designate output directory for created script.
output_directory=/data/user/${USER}/storage/moon_pointing_analysis/monte_carlo/11069
# (3) report output location and name
report_directory=/data/user/${USER}/storage/nohup_reports/
report_name=MC
# (4) designate the feature keys to extract; found via investigating the I3-files using dataio-shovel in IceTray.
keys=(TWSRTHVInIcePulses SplitInIcePulses InIceDSTPulses SplineMPE SplineMPEIC RNNReco RNNReco_sigma )


## do not alter beyond this point ##
# date for report name
TIMESTAMP=$(date "+%H%M%S")

# if directories does not exist for reporting, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# save the report file to 
report_location=${report_directory}${report_name}${TIMESTAMP}.out

# run script
nohup python $(readlink -f .)/moon_pointing_analysis/data/convert_i3_files.py \
--db ${database_directory} \
--outdir ${output_directory} \
--keys ${keys} \
> ${report_location}