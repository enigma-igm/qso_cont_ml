'''Utility functions for the evaluation procedures.'''

import numpy as np
#from numba import jit
from IPython import embed

#@jit
def bootstrapMean(data, iterations=100):

    # draw [iterations] random samples from data and compute the mean spectrum over this set
    n_qso = data.shape[0]
    n_wav = data.shape[-1]
    rng = np.random.default_rng()
    idcs = rng.integers(0, n_qso, size=(iterations, n_qso))

    means = np.zeros((iterations, n_wav))

    print ("Shape of data:", data.shape)

    for iteration, idcs_it in enumerate(idcs):

        data_it = data[idcs_it]
        print ("Shape of data_it:", data_it.shape)
        mean_it = np.mean(data_it, axis=0)

        if isinstance(mean_it, float):
            embed()
        else:
            means[iteration] = mean_it

    sigma_min, sigma_plus = np.percentile(means, [16.,84.])

    return sigma_min, sigma_plus