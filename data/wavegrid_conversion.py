import numpy as np
from qso_fitting.data.utils import rebin_spectra, get_wave_grid
import torch
from data.load_datasets import Spectra
from data.load_data import normalise_spectra
from torch.utils.data import Dataset
import astropy.constants as const

class InputSpectra(Dataset):
    def __init__(self, wave_grid, flux, ivar, redshifts, restframe=True, wave_min=1000., wave_max=1970., dloglam=1.e-4,
                 cont=None):
        """
        Class for storing data we wish to get continuum predictions for.
        The spectra are regridded onto the native rest-frame wavelength grid of the network.
        If the given wavelengths are observed wavelengths, they are converted to rest-frame wavelengths first.

        @param wave_grid: ndarray of shape (n_qso, n_wav)
                Wavelength grid of the data. Either rest-frame or observer-frame, as specified by parameter 'restframe'.
        @param flux: ndarray of shape (n_qso, n_wav)
                Spectra of the quasars.
        @param ivar: ndarray of shape (n_qso, n_wav)
                Inverse variance vectors of the spectra.
        @param redshifts: ndarray of shape (n_qso,) or float
                Redshifts of the quasars. If a single float is given, it is assumed to be the redshift of all quasars.
        @param restframe: bool
                If True, assumes the wavelength grids are in rest-frame wavelengths.
                If False, the observed wavelengths are converted to rest-frame wavelengths.
        @param wave_min: float
                Minimum wavelength of the network grid.
        @param wave_max: float
                Maximum wavelength of the network grid.
        @param dloglam: float
                log(wav) spacing of the network wavelength grid.
        @param cont: NoneType or ndarray of shape (n_qso, n_wav)
                True continuum of the quasars, if available.
        """

        self.n_qso = flux.shape[0]

        if isinstance(redshifts, float):
            self.redshifts = np.full(self.n_qso, redshifts)
        elif isinstance(redshifts, np.ndarray):
            if len(redshifts) == self.n_qso:
                self.redshifts = redshifts
        else:
            raise TypeError("Parameter 'redshifts' must be either a float or an array of shape (n_qso,).")

        self.flux_orig = flux
        self.ivar_orig = ivar

        # convert to rest-frame wavelengths if necessary
        if not restframe:
            self.wave_rest_orig = wave_grid / (1 + redshifts)
        else:
            self.wave_rest_orig = wave_grid
            
        # normalise the flux and noise
        noise = 1 / np.sqrt(ivar)
        flux_norm, noise_norm = normalise_spectra(self.wave_rest_orig, flux, noise)
        ivar_norm = 1 / noise_norm**2

        if cont is not None:
            _, cont_norm = normalise_spectra(self.wave_rest_orig, flux, cont)

        # construct the native network grid
        self.wave_min = wave_min
        self.wave_max = wave_max
        c_light = (const.c.to("km/s")).value
        self.dvpix = dloglam * c_light * np.log(10)
        self.wave_rest = get_wave_grid(self.wave_min, self.wave_max, self.dvpix)

        # regrid onto the network grid
        self.flux, self.ivar, self.gpm, self.count_rebin = rebin_spectra(self.wave_rest, self.wave_rest_orig,
                                                                         flux_norm, ivar_norm)

        if cont is not None:
            ivar_cont = np.full(cont_norm.shape, 1e20)
            self.cont, _, _, _ = rebin_spectra(self.wave_rest, self.wave_rest_orig, cont_norm, ivar_cont)
            self.cont = torch.FloatTensor(self.cont)

        # make torch tensors of everything
        self.flux = torch.FloatTensor(self.flux)
        self.ivar = torch.FloatTensor(self.ivar)
        self.redshifts = torch.FloatTensor(self.redshifts)
        self.gpm = torch.tensor(self.gpm)

        print ("Regridded the spectra.")
        

    def __len__(self):
        return len(self.flux)


    def __getitem__(self, idx):
        """

        @param idx: int
                Index of the item to return
        @return:
            flux: torch.tensor
            ivar: torch.tensor
            redshift: torch.tensor
            gpm: torch.tensor
        """

        flux = self.flux[idx]
        ivar = self.ivar[idx]
        redshift = self.redshifts[idx]
        gpm = self.gpm[idx]

        try:
            cont = self.cont[idx]
            return flux, ivar, redshift, gpm, cont

        except:
            return flux, ivar, redshift, gpm


    def add_channel_shape(self, n_channels=1):
        '''TO DO: also change redshift shape once we add it to the network input'''

        reshaped_specs = []

        for matrix in [self.flux, self.ivar]:
            matrix = matrix.reshape((len(matrix), n_channels, matrix.shape[1]))
            reshaped_specs.append(matrix)

        self.flux = reshaped_specs[0]
        self.ivar = reshaped_specs[1]