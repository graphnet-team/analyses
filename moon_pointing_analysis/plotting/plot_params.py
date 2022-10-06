"""Standardizing plotting style across all plots"""

SUPTITLE_SIZE = 16
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 16
LEGEND_SIZE = 15
DOT_SIZE = 24

# plot size based on subplots
#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True
single = (12,9)
double = (12,5)
triple = (12,3)

# ylim buffer
size = 0.02

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
    
plt.rc('xtick', labelsize=TICK_SIZE)
plt.rc('ytick', labelsize=TICK_SIZE)
plt.rc('axes', labelsize=LABEL_SIZE,titlesize=TITLE_SIZE)
plt.rc('legend',fontsize=LEGEND_SIZE)
plt.rcParams['legend.title_fontsize'] = LEGEND_SIZE