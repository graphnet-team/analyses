
# MC data
#db=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/09000-09999/
#gcd=/mnt/lfs6/sim/IceCube/2012/filtered/level2/neutrino-generator/11069/08000-08999/GeoCalibDetectorStatus_2012.56063_V1.i3.gz

# MC data; local
db=/scratch/mholm/database/I3files/
gcd=/scratch/mholm/database/I3files/GeoCalibDetectorStatus_2012.56063_V1.i3.gz

pulse="InIceDSTPulses"

outdir=/scratch/mholm/database/
report_location=/data/user/mholm/data/nohup_reports/MCL21.out

nohup python /data/user/mholm/graphnet/studies/Moon_Pointing_Analysis/data/convert_i3_files.py \
--db ${db} \
--gcd ${gcd} \
--pulse ${pulse} \
--outdir ${outdir} \
> ${report_location}