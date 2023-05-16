#!/usr/bin/env bash
# Description: <this is a template, add relevant description here>

# to recreate the results, follow the steps below by designating;
# (1) directory containing I3-files, including gcd file.
database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/...
# (2) output directory for created script.
output_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/...
# (3) report output location and name
report_directory=/groups/icecube/${USER}/storage/nohup_reports/
report_name=<name of report>
# (4) pulsemaps to extract using featureextractor; found via investigating the I3-files using dataio-shovel in IceTray.
pulsemaps=(pulse1, pulse2, ..., pulsen)
# (5) run shell in terminal using "bash <path_to_file>.sh"


## do not alter beyond this point ##
# date for report name
TIMESTAMP=$(date "+%H%M%S")

# if directories does not exist for reporting and output, creates them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# save the report file to:
report_location=${report_directory}${report_name}${TIMESTAMP}.out

# Starts IceTray
eval `/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/setup.sh`
/cvmfs/icecube.opensciencegrid.org/py3-v4.1.0/RHEL_7_x86_64/metaprojects/combo/stable/env-shell.sh

# run script
nohup python $(dirname -- "$(readlink -f "${BASH_SOURCE}")")/convert_i3_files.py & \
--db ${database_directory} \
--out ${output_directory} \
--keys ${keys[@]} \
> ${report_location}

# exit IceTray
exit