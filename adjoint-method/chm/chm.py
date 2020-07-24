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
Lx, Ly = (20., 20.)
Nx, Ny = (256, 256)
Beta = 1.0
Viscosity = 1e-3
Friction = 1e-3

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
amp_total = np.sum(amp_random**2)
# On average, forcing should be at constant density
amp_random = amp_random / amp_total * Lx * Ly

rng = np.random.default_rng(seed=42*rank)

forced_r = False
forced_i = False

def forcing_r(t, deltaT):
    global forced_r
    cshape = domain.dist.coeff_layout.local_shape(scales=1)
    cslice = domain.dist.coeff_layout.slices(scales=1)

    if t < 19.02:
        return np.zeros(cshape)
    elif not forced_r:
        forced_r = True
        ny_grid, nx_grid = np.meshgrid(ny_global, nx_global)
        delta = np.logical_and(ny_grid[cslice]==10, nx_grid[cslice]==0)

        force = delta / deltaT

        return force
    else:
        return np.zeros(cshape)

def forcing_i(t, deltaT):
    global forced_i
    cshape = domain.dist.coeff_layout.local_shape(scales=1)
    cslice = domain.dist.coeff_layout.slices(scales=1)

    if t < 19.02:
        return np.zeros(cshape)
    elif not forced_i:
        forced_i = True
        ny_grid, nx_grid = np.meshgrid(ny_global, nx_global)
        delta = np.logical_and(ny_grid[cslice]==10, nx_grid[cslice]==0)

        force = 1j * delta / deltaT

        return force
    else:
        return np.zeros(cshape)

forcing_func_r = operators.GeneralFunction(domain, 'c', forcing_r, args=[])
forcing_func_i = operators.GeneralFunction(domain, 'c', forcing_i, args=[])

# Set up problem equations
problem = de.IVP(domain, variables=['psi', 'vx', 'vy', 'q', 'psi_r', 'q_r', 'psi_i', 'q_i'])
problem.parameters['Bt'] = Beta
problem.parameters['Mu'] = Viscosity
problem.parameters['Al'] = Friction
problem.parameters['Ly'] = Ly
problem.parameters['Lx'] = Lx
problem.parameters['forcing_func_r'] = forcing_func_r
problem.parameters['forcing_func_i'] = forcing_func_i
problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A))"
problem.substitutions['pb(A,B)'] = "dx(A)*dy(B) - dy(A)*dx(B)"

problem.add_equation("dt(q) + Mu*Lap(Lap(q)) + Al*q - Bt*dy(psi) = -(vx*dx(q) + vy*dy(q))")

problem.add_equation("q - Lap(psi) + psi - integ(psi,'y')/Ly = 0", condition="(nx!=0) or (ny!=0)")
problem.add_equation("psi = 0", condition="(nx==0) and (ny==0)")
problem.add_equation("vy - dx(psi) = 0")
problem.add_equation("vx + dy(psi) = 0")

problem.add_equation("dt(q_r) + Mu*Lap(Lap(q_r)) + Al*q_r - Bt*dy(psi_r) = -(vx*dx(q_r) + vy*dy(q_r)) - pb(psi_r,q) + forcing_func_r")
problem.add_equation("q_r - Lap(psi_r) + psi_r - integ(psi_r,'y')/Ly = 0", condition="(nx!=0) or (ny!=0)")
problem.add_equation("psi_r = 0", condition="(nx==0) and (ny==0)")
problem.add_equation("dt(q_i) + Mu*Lap(Lap(q_i)) + Al*q_i - Bt*dy(psi_i) = -(vx*dx(q_i) + vy*dy(q_i)) - pb(psi_i,q) + forcing_func_i")
problem.add_equation("q_i - Lap(psi_i) + psi_i - integ(psi_i,'y')/Ly = 0", condition="(nx!=0) or (ny!=0)")
problem.add_equation("psi_i = 0", condition="(nx==0) and (ny==0)")



# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

timestep = 2e-5
max_timestep = 0.2
#snapshotStep = 0.0005
snapshotStep = 0.02


# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():
    # Set up initial conditions
    q = solver.state['q']

    # Random perturbations, initialized globally for same results in parallel
    cshape = domain.dist.coeff_layout.local_shape(scales=1)
    cslice = domain.dist.coeff_layout.slices(scales=1)

    noise = rng.standard_normal(cshape)

    base = noise*amp_random[cslice]*0.25

    q['c'] = base

    # Timestepping and output
    dt = timestep
    stop_sim_time = 20.0
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
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.7,
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
        forcing_func_r.args = [solver.sim_time, dt]
        forcing_func_i.args = [solver.sim_time, dt]
        dt = solver.step(dt)
        if (solver.iteration-1) % output_cadence == 0:
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
