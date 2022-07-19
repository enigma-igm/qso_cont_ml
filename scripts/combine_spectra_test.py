'''Test script for creating a file containing spectra from two simulators.'''

from simulator.singular import FullSimulator
from simulator.combined import CombinedSimulations

z1 = 2.8
z2 = 3.5
mag = 18.5
nsamp_each = 1000

sim1 = FullSimulator(nsamp_each, z1, mag)
sim2 = FullSimulator(nsamp_each, z2, mag)
sims_list = [sim1, sim2]

combined_sims = CombinedSimulations(sims_list)
combined_sims.saveFile()