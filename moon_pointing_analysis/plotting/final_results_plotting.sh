# data pathing
indir = "/groups/icecube/qgf305/storage/MoonPointing/Models/inference/Sschindler_data_L4/Merged_database/"
outdir = "/groups/icecube/qgf305/work/graphnet/studies/Moon_Pointing_Analysis/plotting/reconstruction/test_plot/"

# script executions
python reconstruction/plot_reconstruction.py --db ${indir} --o ${outdir}