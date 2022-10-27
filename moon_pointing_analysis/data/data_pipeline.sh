#!/usr/bin/env bash
# MC data
#db=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/09000-09999/
#gcd=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/08000-08999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz

# database; local
db=/data/user/sschindler/zeuthenCluster_moonL4_exp13_01

# pulse type
pulse=InIceDSTPulses

report_name=RD

## do not alter beyond this point
# date for report name
TIMESTAMP=$(date "+%H%M%S")

outdir=/data/user/pa000/MoonPointing/Sschindler_data_generic_extraction
report_location=/data/user/${USER}/MoonPointing/Reports/${report_name}${TIMESTAMP}.out

nohup python /home/${USER}/analyses/moon_pointing_analysis/data/convert_i3_files.py \
--db ${db} \
--pulse ${pulse} \
--outdir ${outdir} \
> ${report_location}