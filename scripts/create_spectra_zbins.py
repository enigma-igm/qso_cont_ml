'''Script for generating synthetic spectra in a few redshift bins.'''

from simulator.singular import FullSimulator
from simulator.combined import CombinedSimulations

redshifts = [2.5, 2.8, 3.1, 3.4, 3.7]
mag = 18.5
nsamp_each = int(25000 / len(redshifts))

sims_list = [FullSimulator(nsamp_each, z, mag, half_dz=0.05) for z in redshifts]

combined_sims = CombinedSimulations(sims_list)
combined_sims.saveFile()