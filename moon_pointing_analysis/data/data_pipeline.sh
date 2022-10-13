#!/usr/bin/env bash
# MC data
#db=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/09000-09999/
#gcd=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/08000-08999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz

# database; local
db=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/11069/I3files

# pulse type
pulse=InIceDSTPulses

report_name=MC

## do not alter beyond this point
# date for report name
TIMESTAMP=$(date "+%H%M%S")

outdir=/groups/icecube/petersen/GraphNetDatabaseRepository/moon_pointing_analysis/monte_carlo/11069
report_location=/groups/icecube/${USER}/storage/nohup_reports/${report_name}${TIMESTAMP}.out

nohup python /groups/icecube/${USER}/work/analyses/moon_pointing_analysis/data/convert_i3_files.py \
--db ${db} \
--pulse ${pulse} \
--outdir ${outdir} \
> ${report_location}