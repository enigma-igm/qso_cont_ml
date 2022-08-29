from simulator.redshift_loop import simulateInRedshiftLoop

nsamp = 25000
dz = 0.08
copy_factor = 50
combined_sims = simulateInRedshiftLoop(nsamp, dz, copy_factor=copy_factor)