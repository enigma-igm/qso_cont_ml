# This file contains useful measures of error

import numpy as np

def MSE(y_obs, y_pred):
    '''Computes the mean squared error for the vector of observed values y_obs
    and the vector of predicted values y_pred. Both are arrays of shape
    (n_samples, n_features).'''

    diffsq = (y_obs - y_pred)**2
    diffsq_over_features = np.sum(diffsq, axis=1)
    mse = np.mean(diffsq_over_features)
    return mse
