db=/groups/icecube/qgf305/storage/I3_files/Sebastian_MoonL4/
gcd=None
pulse="SRTInIcePulses"

python convert_i3_to_sqlite.py --db ${db} --gcd ${gcd} --pulse ${pulse}