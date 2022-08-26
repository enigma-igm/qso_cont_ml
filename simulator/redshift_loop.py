'''Module for running the simulator in a loop over redshifts, with the aim of creating a representative sample of mock
spectra.'''

import numpy as np
from data.boss_eda.load import loadRedshiftLuminosityFile
from simulator.singular import FullSimulator
from simulator.combined import CombinedSimulations


def simulateInRedshiftLoop(nsamp, dz, savefile=None):

    if savefile is None:
        savefile = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/mockspec_dz{}.hdf5".format(dz)



def createCopyQSOs(nsamp, datapath=None):

    z_data, logLv_data = loadRedshiftLuminosityFile(datapath)

    ncopies = 10 * nsamp
