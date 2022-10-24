#!/usr/bin/env bash
# Description: Contains a small scale test on hep, containing 3 I3-files

# to recreate the results, follow the steps below.
# (1) designate directory containing I3-files, including gcd file.
database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/11069/I3files
# (2) designate output directory for created script.
output_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/11069
# (3) report output location and name
report_directory=/groups/icecube/${USER}/storage/nohup_reports/
report_name=MC
# (4) designate the feature keys to extract; found via investigating the I3-files using dataio-shovel in IceTray.
keys=(I3MCTree, SplitInIcePulses, InIceDSTPulses)


## do not alter beyond this point ##
# date for report name
TIMESTAMP=$(date "+%H%M%S")

# if directories does not exist for reporting, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# save the report file to 
report_location=${report_directory}${report_name}${TIMESTAMP}.out

# Starts IceTray
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh

# run script based on bash script location
nohup python $(dirname -- "$(readlink -f "${BASH_SOURCE}")")/convert_i3_files.py \
--db ${database_directory} \
--out ${output_directory} \
--keys ${keys[@]} \
> ${report_location}