'''Utility functions for the evaluation procedures.'''

import numpy as np
from numba import jit

@jit
def bootstrapMean(data, iterations=100):

    # draw [iterations] random samples from data and compute the mean spectrum over this set
    n_qso = data.shape[0]
    rng = np.random.default_rng()
    idcs = rng.integers(0, n_qso, size=(iterations, n_qso))

    means = np.zeros(iterations)

    for iteration, idcs_it in enumerate(idcs):

        data_it = data[idcs_it]
        means[iteration] = np.mean(data_it, axis=0)

    sigma_min, sigma_plus = np.percentile(means, [16.,84.])

    return sigma_min, sigma_plus