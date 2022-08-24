'''Module for exporting the inferred/binned BOSS DR14 QSO distribution P(z, logLv).'''

import numpy as np
import h5py
from visualisation import RedshiftHistogram, LuminosityHistogram, binEdges
from load import loadRedshiftLuminosityFile

# see conditional_hists_script

class HistogramExporter:
    '''
    Class for creating a marginal redshift histogram and a number of conditional logLv histograms, and exporting them
    to an HDF5 file.

    Attributes:
        dz: float
        dlogLv: float
        SN_min: float
        z_bin_edges: ndarray
        logLv_edges: ndarray
        n_z_bins: int
        n_lum_bins: int
        z_hist: RedshiftHistogram instance
        lum_hists: list of LuminosityHistogram instances
    '''

    def __init__(self, dz, dlogLv, datapath=None, SN_min=10):

        # load the data
        redshifts, logLv = loadRedshiftLuminosityFile(datapath)

        self.dz = dz
        self.dlogLv = dlogLv
        self.SN_min = SN_min

        # set the bin edges and limits on logLv (due to outliers)
        self.z_bin_edges = binEdges(redshifts, dz)
        self.logLv_edges = binEdges(logLv, dlogLv, data_range=np.percentile(logLv, [0.1, 99.9]))
        logLv_lims = np.percentile(logLv, [0.1, 99.9])

        self.n_z_bins = self.z_bin_edges.size - 1
        self.n_lum_bins = self.logLv_edges.size - 1

        # create a marginal histogram of redshifts
        self.z_hist = RedshiftHistogram(redshifts, logLv, logLv_lims=logLv_lims, bins=self.z_bin_edges)

        # create conditional histograms of log-luminosity
        self.lum_hists = []

        for i in range(self.n_z_bins):

            lum_hist_i = LuminosityHistogram(logLv, redshifts, redshift_lims=self.z_bin_edges[i:i+2],
                                             range=logLv_lims, bins=self.logLv_edges)
            self.lum_hists.append(lum_hist_i)


    def saveToFile(self, savepath=None):
        '''
        Save the histograms to a single HDF5 file.

        @param savepath: NoneType or str
        @return:
        '''

        if savepath is None:
            savepath = "/net/vdesk/data2/buiten/MRP2/Data/"

        filename = "BOSS-luminosity-redshift-histograms.hdf5"

        f = h5py.File(savepath + filename, "w")

        z_hist_data = np.rec.array([self.z_hist.mids, self.z_hist.counts], dtype=[("mids", "<f8"), ("counts", "<i8")])
        z_hist_dset = f.create_dataset("redshift-hist-marginal", data=z_hist_data)
        z_hist_dset.attrs["redshift-width"] = self.dz
        z_hist_dset.attrs["SN_min"] = self.SN_min

        grp_lum = f.create_group("luminosity")
        grp_lum.attrs["logL-width"] = self.dlogLv
        grp_lum.attrs["n-lum-bins"] = self.n_lum_bins

        for i in range(self.n_z_bins):

            lum_hist_i = self.lum_hists[i]
            lum_hist_data = np.rec.array([lum_hist_i.mids, lum_hist_i.counts],
                                         dtype=[("mids", "<f8"), ("counts", "<i8")])
            lum_hist_dset = grp_lum.create_dataset("logLv-hist-conditional-z{}".format(self.z_hist.mids[i]),
                                                   data=lum_hist_data)
            lum_hist_dset.attrs["redshift"] = self.z_hist.mids[i]
            lum_hist_dset.attrs["redshift-width"] = self.dz
            lum_hist_dset.attrs["logL-width"] = self.dlogLv

        f.close()
        print ("File created at {}".format(savepath))


    # TODO: write functions for plotting the histograms