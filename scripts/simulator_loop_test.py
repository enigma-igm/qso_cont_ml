from simulator.redshift_loop import simulateInRedshiftLoop

nsamp = 25000
dz = 0.08
copy_factor = 50

savepath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
combined_sims = simulateInRedshiftLoop(nsamp, dz, copy_factor=copy_factor, savepath=savepath)