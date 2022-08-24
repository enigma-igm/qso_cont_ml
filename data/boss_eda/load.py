'''Module for loading previously extracted BOSS DR14 redshifts and logLv values.'''

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def loadRedshiftLuminosityFile(savepath=None):
    '''
    Load the redshifts and log-luminosities at the Lyman limit of BOSS DR14 quasars.

    @param savepath: str or NoneType
    @return:
    '''

    if savepath is None:
        savepath = os.getenv("SPECDB")
    elif ~isinstance(savepath, str):
        raise TypeError("Argument 'savepath' must be a string or None.")

    f = h5py.File(savepath+"/luminosity-redshift-metadata.hdf5", "r")

    redshifts = np.array(f["redshifts"])
    logLv = np.array(f["logLv"])

    f.close()

    return redshifts, logLv


class HistogramImporter:
    '''
    Class for importing previously created histograms of redshift and luminosity.

    Attributes:
        z_mids:
        z_counts:
        dz:
        dlogLv:
        n_z_bins:
        n_lum_bins:
        lum_hists: ndarray of shape (n_z_bins, n_lum_bins, 2)
            3D array containing the log-luminosity midpoints (final index 0) and corresponding counts (final index 1)
            for each redshift bin.
    '''

    def __init__(self, datafile=None):

        if datafile is None:
            folder = "/net/vdesk/data2/buiten/MRP2/Data/"
            datafile = folder + "BOSS-luminosity-redshift-histograms.hdf5"

        f = h5py.File(datafile, "r")

        self.z_mids= np.array(f["redshift-hist-marginal"]["mids"])
        self.z_counts = np.array(f["redshift-hist-marginal"]["counts"])
        self.dz = f["redshift-hist-marginal"].attrs["redshift-width"]
        self.dlogLv = f["luminosity"].attrs["logL-width"]
        self.n_z_bins = self.z_mids.size
        self.n_lum_bins = f["luminosity"].attrs["n-lum-bins"]

        self.lum_hists = np.zeros((self.n_z_bins, self.n_lum_bins, 2))

        for i in range(self.n_z_bins):

            lum_mids = np.array(f["luminosity/logLv-hist-conditional-z{}".format(self.z_mids[i])]["mids"])
            lum_counts = np.array(f["luminosity/logLv-hist-conditional-z{}".format(self.z_mids[i])]["counts"])

            self.lum_hists[i,:,0] = lum_mids
            self.lum_hists[i,:,1] = lum_counts

        print ("Loaded histogram data.")

        f.close()


    def plotRedshiftHistogram(self, dpi=320):

        fig, ax = plt.subplots(dpi=dpi)

        ax.bar(self.z_mids, self.z_counts, width=self.dz, alpha=.8)
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Occurrences")

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        return fig, ax