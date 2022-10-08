# Several scripts to convert SPICE data into data usable for training

Most notably, `tilt.py` interpolates icedata from the measuring positions to all node positions, provides 2d and 3d representations of it, and builds a lookup table for easy accessing during training.

`loss_weight.py` creates a weight for each event in order to focus on the energy region from 0 to 1.5 log10 GeV.

`energy-dom_count.py` creates a heat map showing the number of DOMs per event against the energy of that event.

`rde.py` builds a lookup table of the relative DOM efficiency.
