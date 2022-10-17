#!/usr/bin/env bash
# Notes: <this is a template, add relevant description here>

# to recreate the results, follow the steps below by designating;
# (1) directory containing I3-files, including gcd file.
database_directory=/groups/icecube/petersen/GraphNetDatabaseRepository/...
# (2) output directory for created script.
output_directory=/groups/icecube/petersen/GraphNetDatabaseRepository...
# (3) report output location and name
report_directory=/groups/icecube/${USER}/storage/nohup_reports/
report_name=<name of report>
# (4) designate the feature keys to extract; found via investigating the I3-files using dataio-shovel in IceTray.
keys=(key1 key2 ... keyn)


## do not alter beyond this point ##
# date for report name
TIMESTAMP=$(date "+%H%M%S")

# if directories does not exist for reporting, create them.
mkdir -p ${output_directory};
mkdir -p ${report_directory};

# pending testing from the above dir creation; the above code should not override... hopefully
#if [ ! -d dir ] 
#then
#    mkdir -p dir;
#fi

# save the report file to 
report_location=${report_directory}${report_name}${TIMESTAMP}.out

# run script
nohup python $(readlink -f .)/moon_pointing_analysis/data/convert_i3_files.py \
--db ${database_directory} \
--outdir ${output_directory} \
--keys ${keys} \
> ${report_location}