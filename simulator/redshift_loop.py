'''Module for running the simulator in a loop over redshifts, with the aim of creating a representative sample of mock
spectra.'''

import numpy as np
from data.boss_eda.load import loadRedshiftLuminosityFile
from simulator.singular import FullSimulator
from simulator.combined import CombinedSimulations


def simulateInRedshiftLoop(nsamp, dz, datapath=None, savefile=None, copy_factor=10):

    if savefile is None:
        savefile = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/mockspec_dz{}.hdf5".format(dz)

    z_data, logLv_data = loadRedshiftLuminosityFile(datapath)
    z_copies, logLv_copies = createCopyQSOs(z_data, logLv_data, copy_factor)
    z_draw, logLv_draw = drawFromCopies(z_copies, logLv_copies, nsamp)





def createCopyQSOs(z_data, logLv_data, copy_factor=10):
    '''
    Creates copy_factor * nsamp copies of the given (redshift, log-luminosity) pairs.

    @param z_data: ndarray of shape (nsamp,)
    @param logLv_data: ndarray of shape (nsamp,)
    @param copy_factor: int
    @return:
        copied_redshifts: ndarray of shape (copy_factor * nsamp,)
        copied_logLv: ndarray of shape (copy_factor * nsamp,)
    '''

    assert len(z_data) == len(logLv_data)
    assert isinstance(copy_factor, int)

    nsamp = len(z_data)
    ncopies = copy_factor * nsamp
    copied_redshifts = np.zeros(nsamp * ncopies)
    copied_logLv = np.zeros(nsamp * ncopies)

    for i, (z, logLv) in enumerate(zip(z_data, logLv_data)):

        copied_redshifts[i:i+ncopies] = np.full(ncopies, z)
        copied_logLv[i:i+ncopies] = np.full(ncopies, logLv)

    return copied_redshifts, copied_logLv


def drawFromCopies(z_copies, logLv_copies, nsamp):
    '''
    Draw nsamp random (z, logLv) pairs from the given copies.

    @param z_copies:
    @param logLv_copies:
    @param nsamp:
    @return:
        z_draw:
        logLv_draw:
    '''

    assert len(z_copies) == len(logLv_copies)

    rng = np.random.default_rng()

    n_choice = len(z_copies)
    idcs = rng.integers(0, n_choice, size=nsamp)

    z_draw = z_copies[idcs]
    logLv_draw = logLv_copies[idcs]

    return z_draw, logLv_draw