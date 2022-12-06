"""
Standardizing plotting style across all plots
Note: from test, variables need to be included first, then imports and functions.
"""

SUPTITLE_SIZE = 16
TITLE_SIZE = 16
LABEL_SIZE = 16
TICK_SIZE = 16
LEGEND_SIZE = 15
DOT_SIZE = 24

# plot size based on subplots
single = (16, 10)
double = (16, 7)
triple = (16, 4)

# ylim buffer for more clear graphs
buffer = 0.02

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("seaborn-white")

plt.rc("xtick", labelsize=TICK_SIZE)
plt.rc("ytick", labelsize=TICK_SIZE)
plt.rc("axes", labelsize=LABEL_SIZE, titlesize=TITLE_SIZE)
plt.rc("legend", fontsize=LEGEND_SIZE)
plt.rcParams["legend.title_fontsize"] = LEGEND_SIZE
plt.rcParams["figure.autolayout"] = True
# plt.rcParams["figure.figsize"] = [7.50, 3.50]


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

class display(object):
    """Display HTML representation of multiple objects in jupyter
        Use as display('object1', 'object2', ...) """
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""

    def __init__(self, *args):
        self.args = args

    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                     for a in self.args)

    def __repr__(self):
       return '\n\n'.join(a + '\n' + repr(eval(a))
                       for a in self.args)