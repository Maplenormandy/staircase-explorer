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
from colorsys import hls_to_rgb
plt.ioff()
#from dedalus.extras import plot_tools


def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg+np.pi) / (2*np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8
    c = np.vectorize(hls_to_rgb)(h,l,s)
    c = np.array(c)
    c = c.swapaxes(0,2)

    return c

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

            fig = plt.figure()
            fig, ax = plt.subplots(2,3, sharey='row', figsize=(4.8*2.7, 4.8),
                                   gridspec_kw={'width_ratios':[3,3,1], 'height_ratios':[24,1]})
            for i in range(3):
                ax[1,i].cla()

            # Plot data
            x = file['scales/x']['1.0'][:]
            y = file['scales/y']['1.0'][:]
            psi = file['tasks/psi'][index,:,:]
            q = file['tasks/q'][index,:,:]
            qr = file['tasks/q_r'][index,:,:]
            qi = file['tasks/q_i'][index,:,:]

            psibar = np.average(psi, axis=1)
            psitilde = psi-psibar[:,np.newaxis]
            #psimax = np.max(np.abs(psitilde))

            qbar = np.average(q, axis=1)
            qtilde = q-qbar[:,np.newaxis]
            qmax = np.max(np.abs(qtilde))

            qz = qr + 1j*qi
            qc = colorize(qz)



            cf = ax[0,1].pcolormesh(y, x, qtilde, cmap='viridis', vmin=-qmax, vmax=qmax)
            ax[0,1].set_aspect('equal')
            ax[0,1].set_title('q_tilde')
            fig.colorbar(cf, cax=ax[1,1], orientation='horizontal')

            cf = ax[0,0].pcolormesh(y, x, np.angle(qz), cmap='twilight')
            ax[0,0].set_aspect('equal')
            ax[0,0].set_title('phase of linear response')
            fig.colorbar(cf, cax=ax[1,0], orientation='horizontal')
            ax[0,0].contour(y, x, psi, colors=['black'], linewidths=0.5)
            #ax[0,0].imshow(qc, aspect='equal', origin='lower', extent=[-10.0,10.0,-10.0,10.0])

            ax[0,2].plot(qbar+x, x, label='qbar')
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

