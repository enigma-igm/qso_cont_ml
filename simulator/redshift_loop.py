'''Module for running the simulator in a loop over redshifts, with the aim of creating a representative sample of mock
spectra.'''

import numpy as np
from scipy.interpolate import interp1d
from data.boss_eda.load import loadRedshiftLuminosityFile
from data.boss_eda.visualisation import binEdges, edgesToMidpoints
from simulator.singular import FullSimulator
from simulator.combined import CombinedSimulations
from dw_inference.simulator.utils import find_closest
from IPython import embed


def simulateInRedshiftLoop(nsamp, dz, datapath=None, savepath=None, copy_factor=10, wave_split=1260.,
                           train_frac=0.9, sampleLv=True):
    '''
    Generate mock spectra in a loop over redshift. The number of spectra to generate and the log-luminosity range to
    use in each redshift bin are based on the BOSS DR14 data.

    @param nsamp: int
        Number of spectra to generate in total.
    @param dz: float
        Redshift bin width for discretising redshift space.
    @param datapath: str or NoneType
        Path where the (z, logLv) data is stored.
    @param savepath: str or NoneType
        Path where the mock spectra are to be stored.
    @param copy_factor: int
        Number of copies to make of each quasar to draw from.
    @param sampleLv: bool
        If True, samples the Lyman limit luminosities from a uniform prior. If False, the sampled luminosities are
        replaced by luminosities from the data.

    @return:
        combined_sims: CombinedSimulations instance
    '''

    # TODO: edit loop such that an equal number of QSOs is generated at each redshift

    z_data, logLv_data = loadRedshiftLuminosityFile(datapath)
    ndata = len(z_data)
    #z_copies, logLv_copies = createCopyQSOs(z_data, logLv_data, copy_factor)
    #z_draw, logLv_draw = drawFromCopies(z_copies, logLv_copies, nsamp)

    # divide the redshift space up in bins and identify the midpoints
    #z_edges = binEdges(z_draw, dz)
    z_edges = binEdges(z_data, dz)
    z_mids = edgesToMidpoints(z_edges)

    print ("z_mids: {}".format(z_mids))

    # assign discrete redshifts to the drawn redshifts
    #closest_idcs = find_closest(z_mids, z_draw)
    #z_draw_use = z_mids[closest_idcs]

    # run the simulation for each discrete redshift
    # TODO: remove sims_list
    sims_filenames_list = []
    #sims_list = []
    logLv_ranges = []

    for i, z in enumerate(z_mids):

        #inbin = (z_draw > z_edges[i]) & (z_draw < z_edges[i+1])
        inbin = (z_data > z_edges[i]) & (z_data < z_edges[i+1])
        ndata_i = int(np.sum(inbin))
        #nsamp_i = int((ndata_i / ndata) * nsamp)

        # generate an equal number of QSOs at each redshift
        # this prevents lower-redshift data from dominating the loss in the network training
        nsamp_i = int(nsamp / len(z_mids))

        print ("Number of samples in data for z = {}: {}".format(np.around(z,2), ndata_i))
        print ("Number of samples to simulate for z = {}: {}".format(np.around(z,2), nsamp_i))

        # the code fails in various places if nsamp < 2
        # TODO: fix bug where z_mids may be a longer list than the list of simulations
        if nsamp_i > 1:
            #logLv_range_i = [logLv_draw[inbin].min(), logLv_draw[inbin].max()]

            # if sampleLv is set to False, use the drawn luminosities to replace the sampled ones in the simulator
            # this is implemented in ProximityWrapper.simulateSpectra
            if sampleLv:
                logLv_use = None

                # use a single, fixed logLv range to create a "rectangular" (z, logLv) grid with full coverage
                logLv_range_i = [logLv_data.min(), logLv_data.max()]
            else:
                logLv_use = inverse_transform_sample1d(logLv_data[inbin], nsamp_i)

                # edit logLv_range_i to reflect the actual range of logLv values used
                logLv_range_i = [logLv_use.min(), logLv_use.max()]

        else:
            # simulate only two samples for this redshift with sampled luminosities
            nsamp_i = 2
            logLv_range_i = [logLv_data.min(), logLv_data.max()]
            logLv_use = None

        logLv_ranges.append(logLv_range_i)

        if (z > 3.6) & (z < 3.7):
            embed()

        # TODO: rewrite such that the single simulator output is saved to a file
        # TODO: let CombinedSimulations load a list of filenames
        sim = FullSimulator(nsamp_i, z, logLv_range_i, half_dz=0.05, wave_split=wave_split, logLv_use=logLv_use)
        file_i = sim.save_simple_file(savepath)
        sims_filenames_list.append(file_i)
        #sims_list.append(sim)

    # combine the simulations and save the mock spectra to an HDF5 file
    #combined_sims = CombinedSimulations(sims_list)
    combined_sims = CombinedSimulations(sims_filenames_list)
    combined_sims.saveFile(savepath, dz=dz, train_frac=train_frac)
    combined_sims.saveTransmissionTemplates(savepath, dz=dz)

    print ("Saved the combined file.")

    print ("logLv ranges used:")
    for logLrange in logLv_ranges:
        print (logLrange)

    return combined_sims



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
    copied_redshifts = np.zeros(ncopies)
    copied_logLv = np.zeros(ncopies)

    for i, (z, logLv) in enumerate(zip(z_data, logLv_data)):

        copied_redshifts[i * copy_factor : (i+1) * copy_factor] = np.full(copy_factor, z)
        copied_logLv[i * copy_factor : (i+1) * copy_factor] = np.full(copy_factor, logLv)

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

    if nsamp < n_choice:

        #idcs = rng.integers(0, n_choice, size=nsamp)
        idcs = rng.choice(n_choice, size=nsamp, replace=False)

    else:
        print ("Warning: number of samples to draw exceeds number of copies to draw from; replacement is used.")
        idcs = rng.choice(n_choice, size=nsamp, replace=True)

    z_draw = z_copies[idcs]
    logLv_draw = logLv_copies[idcs]

    return z_draw, logLv_draw


def inverse_transform_sample1d(data, nsamp):
    '''
    Sample from the distribution of the data using inverse transform sampling.

    @param data (ndarray): 1D array of data to sample from.
    @param nsamp (int): Number of samples to draw.
    @return: samples (ndarray): 1D array of samples.
    '''

    # draw nsamp numbers from a uniform distribution
    rng = np.random.default_rng()
    u = rng.uniform(0, 1, size=nsamp)

    # obtain the cumulative distribution function
    # first sort the data
    data_sort = np.sort(data)
    _cdf = np.cumsum(data_sort)

    # normalise the cdf
    cdf = _cdf / _cdf[-1]

    # invert the cdf
    cdf_inv = interp1d(cdf, data, kind="cubic", fill_value="extrapolate", bounds_error=False)

    # sample from the inverse cdf
    samples = cdf_inv(u)

    return samples