import numpy as np
import h5py
import torch
from torch import FloatTensor
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

class TransmissionTemplates:
    '''
    Class for loading a pre-created transmission template bank from disk.

    Attributes:
        wave_fine:
        wave_hybrid:
        mean_trans_fine:
        mean_trans_hybrid:
        z_mids:
        logLv_mids:

    TODO: write plotting method for visualisation?
    '''

    def __init__(self, filepath, dz):

        assert isinstance(filepath, str)
        assert isinstance(dz, float)

        filename = "{}transmission_templates_dz{}.hdf5".format(filepath, dz)

        f = h5py.File(filename, "r")

        self.wave_fine = FloatTensor(f["/fine-grid/wave-fine"])
        self.wave_hybrid = FloatTensor(f["/hybrid-grid/wave-hybrid"])

        self.mean_trans_fine = FloatTensor(f["/fine-grid/mean-trans"])
        self.mean_trans_hybrid = FloatTensor(f["/hybrid-grid/mean-trans"])

        #self.z_mids = FloatTensor(f["z-mids"])
        #self.logLv_mids = FloatTensor(f["logLv-mids"])
        self.z_mids = np.array(f["z-mids"])
        self.logLv_mids = np.array(f["logLv-mids"])

        self.n_zbins = len(self.z_mids)
        self.n_logLbins = len(self.logLv_mids)

        f.close()


    def _chooseGrid(self, grid="fine"):

        if grid=="fine":
            wave = self.wave_fine.cpu().detach().numpy()
            trans = self.mean_trans_fine.cpu().detach().numpy()

        elif grid=="hyrbid":
            wave = self.wave_hybrid.cpu().detach().numpy()
            trans = self.mean_trans_hybrid.cpu().detach().numpy()

        else:
            raise ValueError("Value for 'grid' not recognised. Use 'fine' or 'hybrid' instead.")

        return wave, trans


    def plotSingle(self, ax, idx_z, idx_logL, grid="fine", alpha=.7):

        wave, _trans = self._chooseGrid(grid)
        trans = _trans[idx_z, idx_logL]

        ax.plot(wave, trans, label=r"z = {}; logL = {}".format(np.round(self.z_mids[idx_z], 1),
                                                               np.round(self.logLv_mids[idx_logL], 1)),
                lw=1.5, alpha=.8)

        ax.legend()
        ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"Mean transmission")

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)

        return ax


    def plotAllRedshifts(self, ax, idx_logL, grid="fine", alpha=.7):

        for i in range(self.n_zbins):
            self.plotSingle(ax, i, idx_logL, grid=grid, alpha=alpha)

        ax.set_title(r"logL = {}".format(np.round(self.logLv_mids[idx_logL], 1)))

        return ax


    def plotAllLuminosities(self, ax, idx_z, grid="fine", alpha=.7):

        for i in range(self.n_logLbins):
            self.plotSingle(ax, idx_z, i, grid=grid, alpha=alpha)

        ax.set_title(r"z = {}".format(np.round(self.z_mids[idx_z], 1)))

        return ax