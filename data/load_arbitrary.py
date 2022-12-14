import numpy as np
import torch
from torch.utils.data import Dataset
from simulator.load_transmission import TransmissionTemplates
import astropy.constants as const
from scipy.interpolate import interp1d

from qso_fitting.data.utils import rebin_spectra, get_wave_grid

try:
    from dw_inference.simulator.utils import get_blu_red_wave_grid
except:
    from utils.grids import get_blu_red_wave_grid

from data.normalisation import normalise_spectra, normalise_ivar

class InputSpectra(Dataset):
    '''
    Class for storing and structuring the spectra for which we wish to get continuum predictions.
    The spectra are regridded onto the hybrid rest-frame wavelength grid of the network.
    An input tensor is created, containing the spectra, corresponding noise vectors (in terms of inverse variance) and
    the mean transmission curves corresponding to the redshift and Lyman-limit luminosity of each quasar.
    These mean transmission curves are determined through bilinear interpolation in (z, logLv) space based on the
    output of the simulator.

    Attributes:
        n_qso: int
        redshifts: ndarray or float
        logLv: ndarray or float
        flux_orig: ndarray
        ivar_orig: ndarray
        wave_rest_orig: ndarray
        wave_min: float
        wave_max: float
        dvpix: float
        wave_fine: ndarray
        wave_hybrid: ndarray
        wave_obs_fine: ndarray
        flux_fine: ndarray
        ivar_fine: ndarray
        gpm_fine: ndarray
        flux_hybrid: FloatTensor
        ivar_hybrid: FloatTensor
        gpm_hybrid: ndarray
        cont_fine: NoneType or ndarray
        cont_hybrid: NoneType or FloatTensor
        mean_trans_hybrid: FloatTensor
        input_hybrid: FloatTensor

    Methods:
        __len__:
        __getitem__:
    '''

    def __init__(self, wave_grid, flux, ivar, redshifts, logLv, restframe=True, wave_min=980., wave_max=1970.,
                 dloglam=1e-4, cont=None, wave_split=1260.,
                 transmission_filepath="/net/vdesk/data2/buiten/MRP2/pca-sdss-old/", transmission_dz=0.08,
                 transmission_nsamp=25000):

        '''

        @param wave_grid: ndarray of shape (n_qso, n_wav)
        @param flux: ndarray of shape (n_qso, n_wav)
        @param ivar: ndarray of shape (n_qso, n_wav)
        @param redshifts: ndarray of shape (n_qso,)
        @param logLv: ndarray of shape (n_qso,)
        @param restframe: bool
        @param wave_min: float
        @param wave_max: float
        @param dloglam: float
        @param cont: NoneType or ndarray of shape (n_qso, n_wav)
        @param wave_split: float
        @param transmission_filepath: str
        @param transmission_dz: float
        '''

        self.n_qso = flux.shape[0]

        if isinstance(redshifts, float):
            self.redshifts = np.full(self.n_qso, redshifts)
        elif isinstance(redshifts, np.ndarray):
            if len(redshifts) == self.n_qso:
                self.redshifts = redshifts
        else:
            raise TypeError("Parameter 'redshifts' must be either a float or an array of shape (n_qso,).")

        if isinstance(logLv, float):
            self.logLv = np.full(self.n_qso, logLv)
        elif isinstance(logLv, np.ndarray):
            if len(logLv) == self.n_qso:
                self.logLv = logLv
        else:
            raise TypeError("Parameter 'logLv' must be either a float or an array of shape (n_qso,).")

        # keep the original flux and ivar noise vectors as attributes
        self.flux_orig = flux
        self.ivar_orig = ivar
        self.wave_split = wave_split

        # convert to rest-frame wavelengths if necessary
        _wave_grid = np.atleast_2d(wave_grid)
        if not restframe:
            self.wave_rest_orig = np.zeros_like(_wave_grid)
            for i in range(len(_wave_grid)):
                self.wave_rest_orig[i] = _wave_grid[i] / (1 + self.redshifts[i])   # can have length 1
        else:
            self.wave_rest_orig = _wave_grid

        # normalise the flux and noise
        flux_norm, ivar_norm = normalise_ivar(self.wave_rest_orig, flux, ivar)

        if cont is not None:
            _, cont_norm = normalise_spectra(self.wave_rest_orig, flux, cont)

        # construct the native uniform BOSS-like grid
        # use nomenclature similar to that of load_data_new.SynthSpectra
        self.wave_min = wave_min
        self.wave_max = wave_max
        c_light = const.c.to("km/s").value
        self.dvpix = dloglam * c_light * np.log(10)
        self.wave_fine = get_wave_grid(self.wave_min, self.wave_max, self.dvpix)

        # construct the hybrid grid native to the network
        dvpix_red = 500.0
        self.wave_hybrid, _, _, _ = get_blu_red_wave_grid(wave_min, wave_max, wave_split, self.dvpix, dvpix_red)

        # also make a 2D wavelength grid
        '''
        self.wave_hybrid2d = np.zeros((self.n_qso, self.wave_hybrid.size))
        for i in range(self.n_qso):
            self.wave_hybrid2d[i] = self.wave_hybrid
        '''

        # create a corresponding observed wavelength grid
        self.wave_obs_fine = np.zeros((self.n_qso, self.wave_fine.size))
        for i in range(self.n_qso):
            self.wave_obs_fine[i] = self.wave_fine * (1 + self.redshifts[i])

        # regrid onto the fine grid
        self.flux_fine, self.ivar_fine, self.gpm_fine, _ = rebin_spectra(self.wave_fine, self.wave_rest_orig, flux_norm,
                                                                         ivar_norm)

        # regrid onto the hybrid grid
        self.flux_hybrid, self.ivar_hybrid, self.gpm_hybrid, _ = rebin_spectra(self.wave_hybrid, self.wave_rest_orig,
                                                                               flux_norm, ivar_norm)

        # interpolate over bad pixels
        # do this only for the hybrid grid, on which the network runs
        _flux_hybrid = np.copy(self.flux_hybrid)
        _ivar_hybrid = np.copy(self.ivar_hybrid)
        for i in range(self.n_qso):

            interpolator_flux = interp1d(self.wave_hybrid[self.gpm_hybrid[i]], self.flux_hybrid[i][self.gpm_hybrid[i]],
                                         kind="cubic", axis=-1, fill_value="extrapolate", bounds_error=False)
            _flux_hybrid[i][~self.gpm_hybrid[i]] = interpolator_flux(self.wave_hybrid[~self.gpm_hybrid[i]])

            goodivar = self.ivar_hybrid[i] > 0
            gpm_ivar = self.gpm_hybrid[i] & goodivar
            interpolator_ivar = interp1d(self.wave_hybrid[gpm_ivar], self.ivar_hybrid[i][gpm_ivar], kind="cubic",
                                         axis=-1, fill_value="extrapolate", bounds_error=False)
            _ivar_hybrid[i][~gpm_ivar] = interpolator_ivar(self.wave_hybrid[~gpm_ivar])

            # set ivar values that are still bad to 1e-4
            bad_ivar = _ivar_hybrid[i] <= 0
            _ivar_hybrid[i][bad_ivar] = 1e-4

        self.flux_hybrid = _flux_hybrid
        self.ivar_hybrid = _ivar_hybrid

        # also do the continuum if it is given
        if cont is not None:
            ivar_cont = np.full(cont_norm.shape, 1e20)
            self.cont_fine, _, self.gpm_cont_fine, _ = rebin_spectra(self.wave_fine, self.wave_rest_orig, cont_norm,
                                                                     ivar_cont)
            self.cont_hybrid, _, self.gpm_cont_hybrid, _ = rebin_spectra(self.wave_hybrid, self.wave_rest_orig,
                                                                         cont_norm, ivar_cont)
            # make a tensor of the hybrid-grid continuum such that it is GPU-compatible
            self.cont_hybrid = torch.FloatTensor(self.cont_hybrid)

        else:
            self.cont_fine = None
            self.cont_hybrid = None

        # estimate mean transmission profiles
        trans_templates = TransmissionTemplates(transmission_filepath, transmission_dz, nsamp=transmission_nsamp)
        self.mean_trans_hybrid = trans_templates.interpolateTransmission(self.redshifts, self.logLv, grid="hybrid")

        # make torch tensors of everything that needs to be GPU-compatible
        self.flux_hybrid = torch.FloatTensor(self.flux_hybrid)
        self.ivar_hybrid = torch.FloatTensor(self.ivar_hybrid)
        self.redshifts = torch.FloatTensor(self.redshifts)
        self.mean_trans_hybrid = torch.FloatTensor(self.mean_trans_hybrid)

        # make a 3D input tensor consisting of the absorption spectrum, the ivar and the mean transmission
        flux_unsq = torch.unsqueeze(self.flux_hybrid, dim=1)
        ivar_unsq = torch.unsqueeze(self.ivar_hybrid, dim=1)
        mean_trans_unsq = torch.unsqueeze(self.mean_trans_hybrid, dim=1)
        self.input_hybrid = torch.cat([flux_unsq, ivar_unsq, mean_trans_unsq], dim=1)


    def __len__(self):
        return self.n_qso


    def __getitem__(self, idx):

        input_hybrid = self.input_hybrid[idx]

        return input_hybrid