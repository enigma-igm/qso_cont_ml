import numpy as np
from qso_fitting.data.utils import rebin_spectra, get_wave_grid
from linetools.lists.linelist import LineList
from dw_inference.simulator.utils import get_blu_red_wave_grid
import torch
from data.load_datasets import Spectra
from data.load_data import normalise_spectra, normalise_ivar
from torch.utils.data import Dataset
import astropy.constants as const
from scipy.interpolate import interp1d
from IPython import embed

class InputSpectra(Dataset):
    def __init__(self, wave_grid, flux, ivar, redshifts, restframe=True, wave_min=980., wave_max=2040., dloglam=1.e-4,
                 cont=None, wave_split=1216.):
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

        '''TO DO: make redshifts 2D on new wavelength grid'''
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
        # do this row-wise
        if not restframe:
            self.wave_rest_orig = np.zeros_like(wave_grid)
            for i in range(len(wave_grid)):
                self.wave_rest_orig[i] = wave_grid[i] / (1 + redshifts[i])
        else:
            self.wave_rest_orig = wave_grid

        if wave_split == 1216.:
            strong_lines = LineList("Strong", verbose=False)
            wave_split = strong_lines["HI 1215"]["wrest"].value
            
        # normalise the flux and noise
        flux_norm, ivar_norm = normalise_ivar(self.wave_rest_orig, flux, ivar)

        if cont is not None:
            _, cont_norm = normalise_spectra(self.wave_rest_orig, flux, cont)

        # construct the native uniform BOSS-like grid
        self.wave_min = wave_min
        self.wave_max = wave_max
        c_light = (const.c.to("km/s")).value
        self.dvpix = dloglam * c_light * np.log(10)
        self.wave_rest = get_wave_grid(self.wave_min, self.wave_max, self.dvpix)

        # construct the hybrid network grid
        dvpix_red = 500.0
        self.wave_grid , _, _, _= get_blu_red_wave_grid(wave_min, wave_max, wave_split, self.dvpix, dvpix_red)

        # also make a 2D wavelength grid to deal with the good pixel mask more easily
        self.wave_grid2d = np.zeros((self.n_qso, self.wave_grid.size))
        for i in range(self.n_qso):
            self.wave_grid2d[i] = self.wave_grid

        # regrid onto the uniform BOSS-like grid
        self.flux_uni, self.ivar_uni, self.gpm_uni, self.count_rebin_uni = rebin_spectra(self.wave_rest,
                                                                                         self.wave_rest_orig, flux_norm,
                                                                                         ivar_norm)

        # regrid onto the hybrid BOSS-like grid
        self.flux, self.ivar, self.gpm, self.count_rebin = rebin_spectra(self.wave_grid, self.wave_rest,
                                                                         self.flux_uni, self.ivar_uni, gpm=self.gpm_uni)

        if cont is not None:
            ivar_cont = np.full(cont_norm.shape, 1e20)
            self.cont, _, self.gpm_cont, _ = rebin_spectra(self.wave_grid, self.wave_rest_orig, cont_norm, ivar_cont)
            self.cont_uni, _, _, _ = rebin_spectra(self.wave_rest, self.wave_rest_orig, cont_norm, ivar_cont)
            self.cont = torch.FloatTensor(self.cont)

        # interpolate for bad pixels
        # needs to be done row-wise
        flux_good = np.copy(self.flux)
        ivar_good = np.copy(self.ivar)

        for i in range(self.n_qso):

            interpolator = interp1d(self.wave_grid[self.gpm[i]], self.flux[i][self.gpm[i]], kind="cubic", axis=-1,
                                    fill_value="extrapolate", bounds_error=False)
            interpolated = interpolator(self.wave_grid[~self.gpm[i]])
            flux_good[i][~self.gpm[i]] = interpolated

            goodivar = self.ivar > 0
            gpm_ivar = self.gpm[i] & goodivar
            interpolator_ivar = interp1d(self.wave_grid[gpm_ivar], self.ivar[i][gpm_ivar], kind="cubic", axis=-1,
                                         fill_value="extrapolate", bounds_error=False)
            ivar_good[i][~gpm_ivar] = interpolator_ivar(self.wave_grid[~gpm_ivar])


        self.flux = flux_good
        self.ivar = ivar_good

        # make a 2D tensor for the redshifts on the new grid
        redshifts2d = np.zeros((self.n_qso, len(self.wave_grid)))
        for i in range(len(self.redshifts)):
            redshifts2d[i,:] = self.redshifts[i]

        # make torch tensors of everything
        self.flux = torch.FloatTensor(self.flux)
        self.ivar = torch.FloatTensor(self.ivar)
        self.redshifts = torch.FloatTensor(self.redshifts)
        self.redshifts2d = torch.FloatTensor(redshifts2d)
        self.gpm = torch.tensor(self.gpm)

        print ("Regridded the spectra.")

        print ("Shape of self.flux:", self.flux.shape)
        

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
            gpm_cont = self.gpm_cont[idx]
            return flux, ivar, redshift, gpm, cont, gpm_cont

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


    def add_noise_channel(self):
        '''
        Add a channel for the noise (ivar) vectors of the spectra.
        @return:
        '''

        self.add_channel_shape()

        if self.ivar is None:
            raise ValueError("No noise vectors provided.")

        else:
            ivar_reshaped = self.ivar.reshape((len(self.ivar), 1, self.ivar.shape[-1]))

            expanded_specs = []
            for spec in [self.flux]:
                expanded_specs.append(torch.cat((spec, ivar_reshaped), dim=1))

            self.flux = expanded_specs[0]