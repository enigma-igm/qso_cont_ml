'''Module for easily generating noiseless synthetic spectra from one Proximity simulator call.'''

import numpy as np
from dw_inference.sdss.utils import get_wave_grid
from dw_inference.simulator.lognormal.lognormal_new import F_onorbe
from dw_inference.simulator.utils import get_blu_red_wave_grid
from linetools.lists.linelist import LineList
from dw_inference.simulator.proximity.proximity import Proximity
from data.load_data import normalise_spectra
from scipy import interpolate
from qso_fitting.data.utils import rebin_spectra
import astropy.constants as const


class ProximityWrapper(Proximity):

    def __init__(self, z_qso, mag, npca=10, nskew=1000, wave_min=1000., wave_max=1970., fwhm=131.4,
                 dloglam=1.0e-4):

        # extract necessary line information and natural constants
        strong_lines = LineList("Strong", verbose=False)
        lya_1216 = strong_lines["HI 1215"]
        lyb_1025 = strong_lines["HI 1025"]
        wave_1216 = lya_1216["wrest"].value
        wave_1025 = lyb_1025["wrest"].value
        c_light = (const.c.to("km/s")).value

        # set up the native wavelength grid
        dvpix = dloglam * c_light * np.log(10)
        wave_rest = get_wave_grid(wave_min, wave_max, dvpix)

        # set up Ly-alpha forest and proximity zone details
        iforest = (wave_rest > wave_1025) & (wave_rest < wave_1216)
        z_lya = wave_rest[iforest] * (1. + z_qso) / wave_1216 - 1.
        mean_flux_z = F_onorbe(z_lya)
        self.true_mean_flux = np.mean(mean_flux_z)
        mean_flux_range = np.clip([self.true_mean_flux - 0.0001, self.true_mean_flux + 0.0001], 0.01, 1.)

        mags = np.full(5, mag)
        self.L_rescale = 1.
        nF = 2
        nlogL = 2
        L_rescale_range = (self.L_rescale - 1e-4, self.L_rescale+1e-4)

        #mean_flux_vec = mean_flux_range[0] + (mean_flux_range[1] - mean_flux_range[0]) * np.arange(nF) / (nF - 1)
        #L_rescale_vec = L_rescale_range[0] + (L_rescale_range[1] - L_rescale_range[1]) * np.arange(nlogL) / (nlogL - 1)

        # set up the file containing the PCA vectors
        pcapath = "/net/vdesk/data2/buiten/MRP2/Data/"
        pcafilename = "COARSE_PCA_150_1000_2000_forest.pkl"
        pcafile = pcapath + pcafilename

        # initialise the Proximity simulator and immediately extract the mean transmission profile
        super(ProximityWrapper, self).__init__(wave_rest, fwhm, z_qso, mags, nskew, mean_flux_range, nF, npca, pcafile,
                                               nlogL=nlogL, L_rescale_range=L_rescale_range)


    def meanTransmissionFromSkewers(self):
        '''
        Extract the mean transmission profile from the model skewers. Currently (iF, iL) = (0,0) is chosen.

        @return:
        '''

        # TODO: improve the indexing in mean_t_prox
        mean_trans_skewers = np.ones(self.nspec)
        mean_trans_skewers[self.ipix_blu] = self.mean_t_prox[0,0]

        return mean_trans_skewers


    def simulateSpectra(self, nsamp=25000, stochastic=False):
        '''
        Simulate noiseless continua and absorption spectra. The absorption spectra contain the Ly-alpha forest and the
        proximity effect (approximately). The spectra are normalised to one at 1280 \AA..

        @param nsamp: int
        @param stochastic: bool
        @return:
            cont_norm: ndarray of shape (nsamp, nspec)
            flux_norm: ndarray of shape (nsamp, nspec)
        '''

        # sample theta
        theta = self.sample_theta(nsamp)

        if not stochastic:
            # manually set the mean flux and luminosity rescale values to the single value we want
            theta[:,0] = self.true_mean_flux
            theta[:,1] = self.L_rescale

        # simulate the noiseless continua and absorption spectra
        cont, flux = self.simulator(theta, replace=(nsamp > self.nskew), ivar=None)

        # normalise the spectra to one at 1280 \AA
        flux_norm, cont_norm = normalise_spectra(self.wave_rest, flux, cont)

        return cont_norm, flux_norm

