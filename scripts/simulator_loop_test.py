from simulator.redshift_loop import simulateInRedshiftLoop

#nsamp = 25000

# with these settings we get a file with 2500 "train" spectra, 1250 "validation" data and 1250 "test" data
nsamp = 5000
train_frac = 0.5
#dz = 0.5
#dz = 0.08
dz = 0.1
copy_factor = 50

savepath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
combined_sims = simulateInRedshiftLoop(nsamp, dz, copy_factor=copy_factor, savepath=savepath, train_frac=train_frac)