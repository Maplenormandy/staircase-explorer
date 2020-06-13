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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from dedalus.extras import plot_tools


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    #scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    fig = plt.figure(figsize=(6.4,4.8))
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            # Plot data
            x = file['scales/x']['1.0'][:]
            y = file['scales/y']['1.0'][:]
            n = file['tasks/n'][index,:,:]
            psi = file['tasks/psi'][index,:,:]

            psibar = np.zeros(psi.shape)
            psibar = np.average(psi, axis=1)[:,np.newaxis]
            psitilde = psi-psibar
            psimax = np.max(np.abs(psitilde))

            #plt.pcolormesh(x, y, psitilde.T, cmap='viridis', vmin=-psimax, vmax=psimax)
            plt.pcolormesh(x, y, psi.T, cmap='viridis')
            plt.colorbar()
            #plt.contour(x, y, psi.T, levels=32, cmap='viridis', linewidths=0.5)
            plt.axis('equal')

            # Add time title
            title = title_func(file['scales/sim_time'][index])
            plt.title(title)
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
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

