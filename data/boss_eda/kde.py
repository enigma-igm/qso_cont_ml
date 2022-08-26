'''Module for using a kernel density estimation (KDE) model to represent the BOSS DR14 QSO population.'''

import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
from load import loadRedshiftLuminosityFile, HistogramImporter

class KDESampler:

    def __init__(self, redshifts, logLv):

        self.redshifts = redshifts
        self.logLv = logLv
        data = np.array([self.redshifts, self.logLv])

        self.kde = gaussian_kde(data)

        # import the data histograms to get the grid
        hist_importer = HistogramImporter()
        self.z_mids = hist_importer.z_mids
        self.logLv_mids = hist_importer.logLv_mids


    def _sample(self, n_samples):

        samples = self.kde.resample(n_samples)

        return samples