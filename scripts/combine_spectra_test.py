'''Test script for creating a file containing spectra from two simulators.'''

from simulator.singular import FullSimulator
from simulator.combined import CombinedSimulations

z1 = 2.8
z2 = 3.5
logLv_range1 = [30.4, 31.7]
logLv_range2 = [30.8, 32.]
nsamp_each = 100

sim1 = FullSimulator(nsamp_each, z1, logLv_range1, half_dz=0.05)
sim2 = FullSimulator(nsamp_each, z2, logLv_range2, half_dz=0.05)
sims_list = [sim1, sim2]

combined_sims = CombinedSimulations(sims_list)
#combined_sims.saveFile()