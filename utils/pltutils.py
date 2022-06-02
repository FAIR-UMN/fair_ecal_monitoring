import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb, to_rgba

#---------------------------------------------------------------------------------------------------
#                                   matplotlib style settings
#---------------------------------------------------------------------------------------------------
plt.rcParams['axes.linewidth'] = 1.4
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['xtick.minor.size'] = 4.0
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['xtick.labelsize'] = 'large'

#---------------------------------------------------------------------------------------------------
#                                   custom user functions for plotting
#---------------------------------------------------------------------------------------------------

def set_custom_alpha(col_, alpha_):
    rgb_ = to_rgba(col_)
    return (col_[0], col_[1], col_[2], alpha_)

def rgb2rgba(col_):
    _ = []
    for c in col_:
        _.append(float(c)/255.0)
    _.append(1.0)
    return tuple(_)

def getNcols(N=3, cmap_='plasma'):
    cmap = plt.get_cmap(cmap_)
    cols = cmap.colors
    arr = []
    for i in range(N):
        arr.append(cols[int(256*float(i)/float(N))])
    return arr

#---------------------------------------------------------------------------------------------------
