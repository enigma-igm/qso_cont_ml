'''Utility functions for the evaluation procedures.'''

import numpy as np
#from numba import jit
from IPython import embed

#@jit
def bootstrapMean(data, iterations=100, interval=68., minval=0.):
    '''
    Compute confidence intervals on the mean of a given data set, using a non-parametric bootstrap algorithm.

    @param data: ndarray of shape (n_qso, n_wav)
    @param iterations: int
    @param interval: float or int
    @param minval: float
    @return: sigma_min, sigma_plus
    '''

    # draw [iterations] random samples from data and compute the mean spectrum over this set
    n_qso = data.shape[0]
    n_wav = data.shape[-1]
    rng = np.random.default_rng()
    idcs = rng.integers(0, n_qso, size=(iterations, n_qso))

    means = np.zeros((iterations, n_wav))

    for iteration, idcs_it in enumerate(idcs):

        data_it = data[idcs_it]

        # mask out values < minval if minval is not None
        # otherwise use regular averaging

        if minval is None:
            means[iteration] = np.mean(data_it, axis=0)
        else:
            means[iteration] = np.ma.array(data_it, mask=(data_it < minval)).mean(axis=0)

    sigma_min, sigma_plus = np.percentile(means, [50. - (interval / 2), 50. + (interval / 2)], axis=0)

    return sigma_min, sigma_plus