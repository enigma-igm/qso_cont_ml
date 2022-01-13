# This file contains useful measures of error

import numpy as np
from astropy.stats import mad_std
from torch import FloatTensor

def MSE(y_obs, y_pred):
    '''Computes the mean squared error for the vector of observed values y_obs
    and the vector of predicted values y_pred. Both are arrays of shape
    (n_samples, n_features).'''

    diffsq = (y_obs - y_pred)**2
    diffsq_over_features = np.sum(diffsq, axis=1)
    mse = np.mean(diffsq_over_features)
    return mse


def relative_residuals(y_obs, y_pred):
    '''Computes the residuals between the observed value and the predicted
    value relative to the observed value. Calculates the mean, standard
    deviation and MAD standard deviation of these relative residuals
    for every point on the wavelength grid.'''

    rel_resid = (y_obs - y_pred)/y_obs
    mean_spec = np.mean(rel_resid, axis=0)
    std_spec = np.std(rel_resid, axis=0)
    mad_std_spec = mad_std(rel_resid, axis=0)

    return rel_resid, mean_spec, std_spec, mad_std_spec


def corr_matrix_relresids(y_obs, y_pred, n_samples):

    rel_resid, mean_spec, std_spec, mad_std_spec = relative_residuals(y_obs, y_pred)

    # try using the median instead of the mean
    diff = rel_resid - mean_spec
    #diff = rel_resid - np.median(rel_resid, axis=0)
    covar_delta = 1/(n_samples-1) * np.matmul(diff.T, diff)
    corr_delta = covar_delta/np.sqrt(np.outer(np.diag(covar_delta), np.diag(covar_delta)))

    return corr_delta


class WavWeights:
    '''Class for calculating normalised weights to use in the loss function
    based on the width of the wavelength pixels.'''

    def __init__(self, wave_grid, power=1):

        self.wave_grid = wave_grid
        wave_widths = wave_grid[1:] - wave_grid[:-1]
        vel_widths = 2.998e10/(wave_grid[:-1]) * wave_widths
        self.vel_widths = np.concatenate((vel_widths, [vel_widths[-1]]))

        self.weights_unnormed = self.vel_widths**power
        self.weights_norm_factor = len(self.weights_unnormed) / np.sum(self.weights_unnormed)
        self.weights_normed = self.weights_unnormed * self.weights_norm_factor


    @property
    def weights(self):

        weights_tensor = FloatTensor(self.weights_normed)

        return weights_tensor

    #@weights.setter
    #def set_weights(se


    @property
    def weights_in_MSE(self):

        sqrt_weights = FloatTensor(np.sqrt(self.weights_normed))

        return sqrt_weights