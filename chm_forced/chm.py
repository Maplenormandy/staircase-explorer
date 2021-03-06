"""
Dedalus script for Balanced Hasegawa-Wakatani equations

From Majda PoP 2018

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.core import operators

import logging
logger = logging.getLogger(__name__)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parameters
Lx, Ly = (30., 30.)
Nx, Ny = (512, 512)
Beta = 1.0
Viscosity = 1e-4
Friction = 1e-4

# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

nx_global = np.array(list(range(Nx//2)))
ny_global = np.array(list(range(Ny//2))+list(range(-Ny//2+1,0)))
ky_global, kx_global = np.meshgrid(2*np.pi*ny_global/Ly, 2*np.pi*nx_global/Lx)
k2_global = kx_global**2+ky_global**2

# Set up random forcing
amp_random = ky_global*(1+k2_global)*np.exp(-k2_global/2)
amp_total = np.sum(amp_random**2)*2
# On average, forcing should be at constant energy density
amp_random = amp_random / amp_total * Lx * Ly

rng = np.random.default_rng()

def forcing(deltaT):
    cshape = domain.dist.coeff_layout.local_shape(scales=1)
    cslice = domain.dist.coeff_layout.slices(scales=1)

    noise_r = rng.standard_normal(cshape)
    noise_i = rng.standard_normal(cshape)
    force = (noise_r+1j*noise_i)*amp_random[cslice]*0.1

    return force/np.sqrt(deltaT)

forcing_func = operators.GeneralFunction(domain, 'c', forcing, args=[])

# Set up problem equations
problem = de.IVP(domain, variables=['psi', 'vx', 'vy', 'q'])
problem.parameters['Bt'] = Beta
problem.parameters['Mu'] = Viscosity
problem.parameters['Al'] = Friction
problem.parameters['Ly'] = Ly
problem.parameters['Lx'] = Lx
problem.parameters['forcing_func'] = forcing_func
problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A))"

problem.add_equation("dt(q) + Mu*Lap(Lap(q)) + Al*q - Bt*dy(psi) = -(vx*dx(q) + vy*dy(q)) + forcing_func")

problem.add_equation("q - Lap(psi) + psi - integ(psi,'y')/Ly = 0", condition="(nx!=0) or (ny!=0)")
problem.add_equation("psi = 0", condition="(nx==0) and (ny==0)")
problem.add_equation("vy - dx(psi) = 0")
problem.add_equation("vx + dy(psi) = 0")



# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

timestep = 2e-5
max_timestep = 0.2
#snapshotStep = 0.0005
snapshotStep = 0.2


# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():
    # Set up initial conditions
    q = solver.state['q']

    # Random perturbations, initialized globally for same results in parallel
    cshape = domain.dist.coeff_layout.local_shape(scales=1)
    cslice = domain.dist.coeff_layout.slices(scales=1)

    noise_r = rng.standard_normal(cshape)
    noise_i = rng.standard_normal(cshape)

    base = (noise_r + 1j*noise_i)*amp_random[cslice]*0.2

    q['c'] = base

    # Timestepping and output
    dt = timestep
    stop_sim_time = 600
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    stop_sim_time = 600
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snapshotStep, max_writes=600, mode=fh_mode)
snapshots.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.5,
                     max_change=1.5, min_change=0.1, max_dt=max_timestep, threshold=0.05)
CFL.add_velocities(('vx', 'vy'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("vx*vx + vy*vy + psi*psi", name='Energy')

curr_time = time.time()

output_cadence = 50

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        forcing_func.args = [dt]
        dt = solver.step(dt)
        if (solver.iteration-2) % output_cadence == 0:
            next_time = time.time()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Average timestep (ms): %f' % ((next_time-curr_time) * 1000.0 / output_cadence))
            logger.info('Max energy density = %f' % np.sqrt(flow.max('Energy')))
            curr_time = next_time
            if not np.isfinite(flow.max('Energy')):
                raise Exception('NaN encountered')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
