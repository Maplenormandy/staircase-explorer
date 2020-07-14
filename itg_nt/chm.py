"""
Dedalus script for two-field ITG equations

From Krommes and Kolesnikov 2004

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

import logging
logger = logging.getLogger(__name__)

MPI

# Parameters
Lx, Ly = (30., 30.)
tau = 0.05
diff = 0.5
chi = 0.5

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Fourier('y', 256, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

# Set up problem equations
problem = de.IVP(domain, variables=['n', 'T', 'psi', 'vx', 'vy', 'vx2', 'vy2'])
problem.parameters['tau'] = tau
problem.parameters['D'] = diff
problem.parameters['X'] = chi
problem.parameters['Ly'] = Ly
problem.parameters['Lx'] = Lx
problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A))"
problem.substitutions['pb(A,B)'] = "dx(A)*dy(B) - dy(A)*dx(B)"
problem.substitutions['conv1(A)'] = "vx*dx(A) + vy*dy(A)"
problem.substitutions['conv2(A)'] = "vx2*dx(A) + vy2*dy(A)"
problem.substitutions['dw(A)'] = "A - integ(A,'y')/Ly"

# Equation 17(a,b), density evolution and temperature evolution
problem.add_equation("dt(n) + D*Lap(Lap(n)) = -conv1(n) - conv2(T)")
problem.add_equation("dt(T) + D*Lap(Lap(T)) = -conv1(T) - conv2(n+T)")
#problem.add_equation("dt(n) + D*Lap(Lap(n)) = -pb(psi + 0.5*tau*Lap(psi), n) - 0.5*tau*pb(Lap(psi) + 0.5*tau*Lap(Lap(psi)), T)")
#problem.add_equation("dt(T) + D*Lap(Lap(T)) = -pb(psi + 0.5*tau*Lap(psi), T) - 0.5*tau*pb(Lap(psi) + 0.5*tau*Lap(Lap(psi)), n+T)")

# Poisson's equation
problem.add_equation("n - dw(psi) + Lap(psi) + 0.5*tau*dw(Lap(psi)) + 0.5*tau*Lap(T) = 0", condition="(nx!=0) or (ny!=0)")
problem.add_equation("psi = 0", condition="(nx==0) and (ny==0)")
problem.add_equation("vy - dx(psi) - 0.5*tau*dx(Lap(psi)) = 0")
problem.add_equation("vx + dy(psi) + 0.5*tau*dy(Lap(psi)) = 0")
problem.add_equation("vy2 - 0.5*tau*Lap(vy) = 0")
problem.add_equation("vx2 - 0.5*tau*Lap(vx) = 0")




# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

timestep = 1e-3
max_timestep = 1e-2
#snapshotStep = 0.0005
snapshotStep = 0.5

np.random.seed(42)

# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():

    # Initial conditions
    x = 2*np.pi*domain.grid(0)/Lx
    y = 2*np.pi*domain.grid(1)/Ly
    tt = solver.state['T']
    n = solver.state['n']

    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]

    # Linear background + perturbations damped at walls
    #q['g'] = noise*1e-2 + np.cos(3*x+y)*4e-1 + np.cos(0.5*y)*1e-1 + np.sin(5*x-2*y)*3e-1 + np.cos(2*x)*2e-1
    base = noise*0.0
    base2 = noise*0.0

    for wave in range(1024):
        k = np.random.randint(-16, 16, size=2)
        phase = np.random.rand()*2*np.pi
        amplitude = np.sqrt(np.random.chisquare(2))*0.1

        if k[1] == 0 or k[0] == 0:
            continue

        if wave%2 == 0:
            base = base + amplitude*np.cos(k[0]*x + k[1]*y + phase)
        else:
            base2 = base2 + amplitude*np.cos(k[0]*x + k[1]*y + phase)

    base = 10*np.exp(-(x**2)) + base
    base2 = 10*np.exp(-(x**2)) + base2

    tt['g'] = base2
    n['g'] = base

    # Timestepping and output
    dt = timestep
    stop_sim_time = 500.0
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    stop_sim_time = 500.0
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snapshotStep, max_writes=600, mode=fh_mode)
snapshots.add_system(solver.state)

output_cadence = 25

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=10,
                     max_change=1.5, min_change=0.1, max_dt=max_timestep, threshold=0.05)
CFL.add_velocities(('vx', 'vy'))
CFL.add_velocities(('vx2', 'vy2'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=output_cadence)
flow.add_property("dw(n)*dw(n) + dw(T)*dw(T)", name='dw_enstrophy')

curr_time = time.time()


# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % output_cadence == 0:
            next_time = time.time()
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Average timestep (ms): %f' % ((next_time-curr_time) * 1000.0 / output_cadence))
            logger.info('Max DW enstrophy density = %f' % np.sqrt(flow.max('dw_enstrophy')))
            curr_time = next_time
            if not np.isfinite(flow.max('dw_enstrophy')):
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
