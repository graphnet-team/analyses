This folder contains all scipts to replicate the results of the work by Morten Holm and Peter Andresen, presented in their master theses.

The plotting folder contains the following plotting scripts:
- Neutrino_selection_and_comparison 
Contains the main analysis of selecting pure neutrinos in data and MC and comparing distributions. 
- Monte_Carlo_results 
Results in Monte Carlo without comparison to OscNext retro reconstructions
- Monte_Carlo_results_against_retro 1 and 2 
Results in Monte Carlo with comparison to OscNext retro reconstructions
- Our_vs_OscNext_selection_comparison
Analysis of the amount (rate) of neutrinos we get, and a comparison of the final MC neutrinos we have in comparison to the MC neutrinos in the OscNext selection (DOES NOT INCLUDE RETRO MUONS AND NOISE)
- OscNext_rates_per_level
replication of the rate of each Particle OscNext has in their selection levels
- Sneaky_muons_analysis
Analysis of the muons that make it into our neutrino selection
- Multiclass_efficiency_plot (not cleaned)
Analysis of the efficiency of our neutrino classifier with respect to energy, angles etc.
- muon_noise_contamination_analysis (not cleaned)
Figures illustrating where MC muons, noise and neutrinos and probable real data muons, noise and neutrinos fall in various distributions
- plotting_MC_distributions (not cleaned)
Neutrinos and two muon samples distributions in a few variables
- event_viewer (not cleaned)
Plots actual event pulsemaps. 
- lvl_3_variables_plotting (not cleaned)
Replication of lvl3 distributions and cuts



The train_models folder contains the scripts used to train the GNN models.

The use_models folder contains the scripts used to apply the models to new data.


The data used in this work is from OscNext and is located at the HEP server. 

See Peter Andresens thesis for more details on how to replicate this work, including data locations and compatible GraphNeT branch. 