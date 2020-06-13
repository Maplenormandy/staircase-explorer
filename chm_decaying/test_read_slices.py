"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools

# %%

h5 = h5py.File(r"/home/normandy/scratch/chm_decaying/snapshots/snapshots_s1.h5", mode='r')

# %%

plt.figure(figsize=(6.4,4.8))
t = h5['scales']['sim_time'][:]
x = h5['scales']['x']['1.0'][:]
y = h5['scales']['y']['1.0'][:]
psi = h5['tasks']['psi'][7,:,:]
q = h5['tasks']['q'][7,:,:]

plt.title('t = ' +str(t[7]))
plt.pcolormesh(x, y, q, cmap='bwr')
plt.colorbar()
plt.contour(x, y, -psi, cmap='bwr')
plt.axis('square')

