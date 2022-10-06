"""Standardizing plotting style across all plots"""

SUPTITLE_SIZE = 16
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 16
LEGEND_SIZE = 15
DOT_SIZE = 24

# plot size based on subplots
single = eval("1,2,figsize=(12,9)")
double = (12,4)
triple = ()

# ylim buffer
size = 0.02

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('seaborn-white')
    
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)
plt.rc('axes', labelsize=LABEL_SIZE,titlesize=TITLE_SIZE)
plt.rc('legend',fontsize=LEGEND_SIZE)
plt.rcParams['legend.title_fontsize'] = LEGEND_SIZE

def colorbar(mappable):
    """sets a proper colorbar to a given sub plot, that contains axes properties"""
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar