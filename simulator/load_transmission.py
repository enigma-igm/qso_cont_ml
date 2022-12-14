import numpy as np
import h5py
import torch
from torch import FloatTensor
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from IPython import embed

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

    Methods:
        _chooseGrid(grid="fine")
        interpolateTransmission(redshifts, logLs, grid="fine")
        plotSingle
        plotAllRedshifts
        plotAllLuminosities
    '''

    def __init__(self, filepath, dz, interpmethod="linear", nsamp=25000):

        assert isinstance(filepath, str)
        assert isinstance(dz, float)

        filename = "{}transmission_templates_dz{}_nsamp{}.hdf5".format(filepath, dz, nsamp)

        print ("Using file {} for transmission templates.".format(filename))

        f = h5py.File(filename, "r")

        self.wave_fine = FloatTensor(f["/fine-grid/wave-fine"])
        self.wave_hybrid = FloatTensor(f["/hybrid-grid/wave-hybrid"])

        #self.mean_trans_fine = FloatTensor(f["/fine-grid/mean-trans"])
        #self.mean_trans_hybrid = FloatTensor(f["/hybrid-grid/mean-trans"])

        #self.z_mids = np.array(f["z-mids"])
        #self.logLv_mids = np.array(f["logLv-mids"])

        # for LinearNDInterpolator we need the redshifts and logLvs that make up the grid to be 1D arrays
        # we need to reshape all of the arrays that go into the interpolator

        _mean_trans_fine = FloatTensor(f["/fine-grid/mean-trans"])
        _mean_trans_hybrid = FloatTensor(f["/hybrid-grid/mean-trans"])

        _z_mids = np.array(f["z-mids"])
        _logLv_mids = np.array(f["logLv-mids"])

        self.n_zbins = len(_z_mids)
        self.n_logLbins = len(_logLv_mids)

        # _z_mids is 1D, but _logLv_mids is 2D because they differ for each redshift bin
        self.z_mids = np.zeros(self.n_zbins * self.n_logLbins)
        self.logLv_mids = np.zeros(self.n_zbins * self.n_logLbins)
        self.mean_trans_fine = np.zeros((self.n_zbins * self.n_logLbins, len(self.wave_fine)))
        self.mean_trans_hybrid = np.zeros((self.n_zbins * self.n_logLbins, len(self.wave_hybrid)))
        for i in range(self.n_zbins):
            self.z_mids[i*self.n_logLbins:(i+1)*self.n_logLbins] = _z_mids[i]
            self.logLv_mids[i*self.n_logLbins:(i+1)*self.n_logLbins] = _logLv_mids[i]
            self.mean_trans_fine[i*self.n_logLbins:(i+1)*self.n_logLbins] = _mean_trans_fine.cpu().detach().numpy()[i]
            self.mean_trans_hybrid[i*self.n_logLbins:(i+1)*self.n_logLbins] = _mean_trans_hybrid.cpu().detach().numpy()[i]

        #self.z_mids = np.full((len(_z_mids), _logLv_mids.shape[-1]), _z_mids).ravel()
        #self.logLv_mids = _logLv_mids.ravel()

        self.n_zpoints = len(self.z_mids)
        self.n_logLpoints = len(self.logLv_mids)

        #self.mean_trans_fine = _mean_trans_fine.cpu().detach().numpy().reshape((self.n_zpoints, len(self.wave_fine)))
        #self.mean_trans_hybrid = _mean_trans_hybrid.cpu().detach().numpy().reshape((self.n_zpoints, len(self.wave_hybrid)))

        f.close()

        # try non-regular grid interpolation because the logLv_mids are not equally spaced in log space
        print ("Using LinearNDInterpolator for transmission templates.")

        try:
            self.interpolator_fine = LinearNDInterpolator((self.z_mids, self.logLv_mids), self.mean_trans_fine,
                                                          fill_value=1.)
        except:
            embed()
        self.interpolator_hybrid = LinearNDInterpolator((self.z_mids, self.logLv_mids), self.mean_trans_hybrid,
                                                        fill_value=1.)

        # initialise RegularGridInterpolator instances for each grid
        #self.interpolator_fine = RegularGridInterpolator((self.z_mids, self.logLv_mids),
        #                                                 self.mean_trans_fine.cpu().detach().numpy(),
        #                                                 method=interpmethod, bounds_error=False, fill_value=None)
        #self.interpolator_hybrid = RegularGridInterpolator((self.z_mids, self.logLv_mids),
        #                                                   self.mean_trans_hybrid.cpu().detach().numpy(),
        #                                                   method=interpmethod, bounds_error=False, fill_value=None)
        print ("Created interpolators with mode: {}".format(interpmethod))


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


    def interpolateTransmission(self, redshifts, logLs, grid="fine"):
        '''
        Interpolate the mean transmission profiles from the templates onto the given redshifts and log-luminosities.
        The interpolated profiles are clipped to the range [0, 1] to prevent unphysical behaviour.

        @param redshifts: float or ndarray of shape (nsamp,)
        @param logLs: float or ndarray of shape (nsamp,)
        @param grid: str
        @return: interp_trans: ndarray of shape (nsamp, nspec)
        '''

        if grid == "fine":
            interpolator = self.interpolator_fine
            nspec = len(self.wave_fine)

        elif grid == "hybrid":
            interpolator = self.interpolator_hybrid
            nspec = len(self.wave_hybrid)

        else:
            raise ValueError("Value for 'grid' not recognised. Use 'fine' or 'hybrid' instead.")

        _redshifts = np.atleast_1d(redshifts)
        _logLs = np.atleast_1d(logLs)
        interp_trans = np.zeros((len(_redshifts), nspec))

        for i, (z, logLv) in enumerate(zip(_redshifts, _logLs)):
            # clip transmission values to be between 0 and 1
            interp_trans[i] = np.clip(interpolator((z, logLv)), 0, 1)

        return interp_trans


    def plotSingle(self, ax, idx_z, idx_logL, grid="fine", alpha=.7):

        wave, _trans = self._chooseGrid(grid)
        trans = _trans[idx_z, idx_logL]

        ax.plot(wave, trans, label=r"z = {}; logL = {}".format(np.round(self.z_mids[idx_z], 1),
                                                               np.round(self.logLv_mids[idx_logL], 1)),
                lw=1.5, alpha=alpha)

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