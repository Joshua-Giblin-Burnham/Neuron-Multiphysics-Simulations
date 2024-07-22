# Initialise
from .patchsim import *

#-------------------------------------External Imports--------------------------------------------
import sys
import os
import numpy as np
from numba import njit
import math
import scipy.optimize
from scipy import special
from scipy import signal

# # Plotting import and settinngs
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib.ticker import MaxNLocator

# linewidth = 5.92765763889 # inch
# plt.rcParams["figure.figsize"] = (1.61*linewidth, linewidth)
# plt.rcParams['figure.dpi'] = 256
# plt.rcParams['font.size'] = 16
# plt.rcParams["font.family"] = "Times New Roman"

# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'Times New Roman'
# plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
# plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'