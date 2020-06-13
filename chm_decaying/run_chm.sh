mpiexec -n 2 python3 chm.py
mpiexec -n 2 python3 -m dedalus merge_procs snapshots
mpiexec -n 2 python3 plot_slices.py snapshots/*.h5
