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
diff = 12.0

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(-Lx/2, Lx/2), dealias=3/2)
y_basis = de.Chebyshev('y', 256, interval=(-Ly/2, Ly/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

yb = domain.grid(1)

# Set up problem equations
problem = de.IVP(domain, variables=['n', 'T', 'psi', 'ny', 'Ty', 'psiy', 'nyy', 'nyyy', 'Tyy', 'Tyyy'])
problem.meta[:]['y']['dirichlet'] = True

ncc = domain.new_field(name='s')
ncc['g'] = 2*np.exp(-yb**2)
ncc.meta['x']['constant'] = True

problem.parameters['s'] = ncc
problem.parameters['tau'] = tau
problem.parameters['D'] = diff
problem.parameters['Ly'] = Ly
problem.parameters['Lx'] = Lx
problem.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A))"
problem.substitutions['LapY(A, Ay)'] = "dx(dx(A)) + dy(Ay)"
problem.substitutions['LapY2(A, Ayy, Ayyy)'] = "dx(dx(dx(dx(A)))) + 2*dx(dx(Ayy)) + dy(Ayyy)"
problem.substitutions['pb(A,B)'] = "dx(A)*dy(B) - dy(A)*dx(B)"
problem.substitutions['dw(A)'] = "A - integ(A,'x')/Lx"

# Equation 17(a,b), density evolution and temperature evolution
#problem.add_equation("dt(n) - D*LapX(nx,n) = -conv1(n) - conv2(T)")
#problem.add_equation("dt(T) - D*LapX(Tx,T) = -conv1(T) - conv2(n+T)")
problem.add_equation("dt(n) + D*LapY2(n,nyy,nyyy) = -pb(psi + 0.5*tau*Lap(psi), n) - 0.5*tau*pb(Lap(psi) + 0.5*tau*Lap(Lap(psi)), T) + s")
problem.add_equation("dt(T) + D*LapY2(T,Tyy,Tyyy) = -pb(psi + 0.5*tau*Lap(psi), T) - 0.5*tau*pb(Lap(psi) + 0.5*tau*Lap(Lap(psi)), n+T) + s")

# Poisson's equation
problem.add_equation("n - dw(psi) + LapY(psi,psiy) + 0.5*tau*dw(LapY(psi,psiy)) + 0.5*tau*LapY(T,Ty) = 0")

# Boundary conditions
def add_dy(f):
    problem.add_equation("{0}y - dy({0}) = 0".format(f))

add_dy('n')
add_dy('ny')
add_dy('nyy')
add_dy('T')
add_dy('Ty')
add_dy('Tyy')
add_dy('psi')

problem.add_bc('left(n) = 0')
problem.add_bc('right(n) = 0')
problem.add_bc('left(T) = 0')
problem.add_bc('right(T) = 0')

problem.add_bc('left(nyy) = 0')
problem.add_bc('right(nyy) = 0')
problem.add_bc('left(Tyy) = 0')
problem.add_bc('right(Tyy) = 0')

problem.add_bc('left(psi) = 0')
problem.add_bc('right(psi) = 0')



# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

timestep = 5e-3
#max_timestep = 1e-3
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
        amplitude = np.sqrt(np.random.chisquare(2))*0.01

        if k[1] == 0 or k[0] == 0:
            continue

        if wave%2 == 0:
            base = base + amplitude*np.cos(k[0]*x + k[1]*y + phase)
        else:
            base2 = base2 + amplitude*np.cos(k[0]*x + k[1]*y + phase)

    base = 10*np.exp(-(y**2)/2.0) + base
    base2 = 10*np.exp(-(y**2)/2.0) + base2

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

output_cadence = 5

# CFL
#CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=10,
#                     max_change=1.5, min_change=0.1, max_dt=max_timestep, threshold=0.05)
#CFL.add_velocities(('vx', 'vy'))
#CFL.add_velocities(('vx2', 'vy2'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=output_cadence)
flow.add_property("dw(n)*dw(n) + dw(T)*dw(T)", name='dw_enstrophy')

curr_time = time.time()


# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        #dt = CFL.compute_dt()
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
