"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
from os import path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ioff()
#from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    #scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)


    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            # Check if plot already exists to not duplicate work
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)

            if path.exists(str(savepath)):
                continue

            fig, ax = plt.subplots(2,3, sharey='row', figsize=(4.8*2.7, 4.8),
                                   gridspec_kw={'width_ratios':[3,3,1], 'height_ratios':[24,1]})

            # Plot data
            x = file['scales/x']['1.0'][:]
            y = file['scales/y']['1.0'][:]
            psi = file['tasks/psi'][index,:,:]
            q = file['tasks/q'][index,:,:]

            psibar = np.average(psi, axis=1)
            psitilde = psi-psibar[:,np.newaxis]
            psimax = np.max(np.abs(psitilde))

            qbar = np.average(q, axis=1)
            qtilde = q-qbar[:,np.newaxis]
            qmax = np.max(np.abs(qtilde))

            cf = ax[0,0].pcolormesh(y, x, psitilde, cmap='viridis', vmin=-psimax, vmax=psimax)
            ax[0,0].set_aspect('equal')
            fig.colorbar(cf, cax=ax[1,0], orientation='horizontal')
            #ax[0].contour(x, y, psi.T, colors=['black'], linewidths=0.5)

            cf = ax[0,1].pcolormesh(y, x, qtilde, cmap='viridis', vmin=-qmax, vmax=qmax)
            ax[0,1].set_aspect('equal')
            fig.colorbar(cf, cax=ax[1,1], orientation='horizontal')

            ax[0,2].plot(qbar, x, label='qbar')
            ax[0,2].plot(psibar, x, label='psibar')
            ax[0,2].legend(loc='lower right')

            # Add time title
            title = title_func(file['scales/sim_time'][index])
            plt.suptitle(title)
            # Save figure
            fig.savefig(str(savepath), dpi=dpi)
            #fig.clear()
            plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

