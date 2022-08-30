'''Module for easily generating noiseless synthetic spectra from one Proximity simulator call.'''

import numpy as np
import h5py
from dw_inference.sdss.utils import get_wave_grid
from dw_inference.simulator.lognormal.lognormal_new import F_onorbe
from dw_inference.simulator.utils import get_blu_red_wave_grid, find_closest
from linetools.lists.linelist import LineList
from dw_inference.simulator.proximity.proximity import Proximity
from data.load_data import normalise_spectra
from scipy.interpolate import interp1d
from qso_fitting.data.utils import rebin_spectra
import astropy.constants as const
from data.empirical_noise import rebinNoiseVectors, interpBadPixels
from simulator.save import constructFile


class ProximityWrapper(Proximity):

    def __init__(self, z_qso, logLv_range, nlogL=10, npca=10, nskew=1000, wave_min=1000., wave_max=1970.,
                 fwhm=131.4, dloglam=1.0e-4):

        # extract necessary line information and natural constants
        strong_lines = LineList("Strong", verbose=False)
        lya_1216 = strong_lines["HI 1215"]
        lyb_1025 = strong_lines["HI 1025"]
        self.wave_1216 = lya_1216["wrest"].value
        wave_1025 = lyb_1025["wrest"].value
        c_light = (const.c.to("km/s")).value

        # set up the native wavelength grid
        self.dvpix = dloglam * c_light * np.log(10)
        wave_rest = get_wave_grid(wave_min, wave_max, self.dvpix)

        # set up Ly-alpha forest and proximity zone details
        iforest = (wave_rest > wave_1025) & (wave_rest < self.wave_1216)
        z_lya = wave_rest[iforest] * (1. + z_qso) / self.wave_1216 - 1.
        mean_flux_z = F_onorbe(z_lya)
        self.true_mean_flux = np.mean(mean_flux_z)
        #mean_flux_range = np.clip([self.true_mean_flux - 0.0001, self.true_mean_flux + 0.0001], 0.01, 1.)
        mean_flux_range = np.clip([self.true_mean_flux, self.true_mean_flux + 0.0001], 0.01, 1.)

        #mags = np.full(5, mag)
        #self.mag = mag

        # compute the central log-luminosity (corresponding to 1 in L_rescale_range)
        L_min = 10 ** logLv_range[0]
        L_max = 10 ** logLv_range[1]
        self.L_mid = (L_max - L_min) / 2
        logLv_mid = np.log10(self.L_mid)

        # compute the fractions of Lv_mid corresponding to logLv_range
        frac_L_min = L_min / self.L_mid
        frac_L_max = L_max / self.L_mid
        L_rescale_range = [frac_L_min, frac_L_max]

        print ("Central logLv: {}".format(logLv_mid))
        print ("L_rescale_range: {}".format(L_rescale_range))

        nF = 2

        #mean_flux_vec = mean_flux_range[0] + (mean_flux_range[1] - mean_flux_range[0]) * np.arange(nF) / (nF - 1)
        #L_rescale_vec = L_rescale_range[0] + (L_rescale_range[1] - L_rescale_range[1]) * np.arange(nlogL) / (nlogL - 1)

        # set up the file containing the PCA vectors
        pcapath = "/net/vdesk/data2/buiten/MRP2/Data/"
        pcafilename = "COARSE_PCA_150_1000_2000_forest.pkl"
        pcafile = pcapath + pcafilename

        # initialise the Proximity simulator and immediately extract the mean transmission profile
        super(ProximityWrapper, self).__init__(wave_rest, fwhm, z_qso, nskew, mean_flux_range, nF, npca, pcafile,
                                               mags=None, logLv=logLv_mid, nlogL=nlogL, L_rescale_range=L_rescale_range)


    def meanTransmissionFromSkewers(self):
        '''
        DEPRECATED; use meanTransmissionFromTheta instead!
        Extract the mean transmission profile from the model skewers. Currently (iF, iL) = (0,0) is chosen.

        @return:
        '''

        # TODO: improve the indexing in mean_t_prox
        # TODO: make mean_t_prox luminosity-dependent by using indexing (iF, iL) using find_closest
        mean_trans_skewers = np.ones(self.nspec)
        mean_trans_skewers[self.ipix_blu] = self.mean_t_prox[0,0]

        return mean_trans_skewers


    def meanTransmissioFromTheta(self, theta):
        '''
        Extract the mean transmission profile for each simulated spectrum from the nearest mean flux and logLv value.

        @param theta: ndarray of shape (nsamp, npca + 4)
            Sampled model parameters. The order is: [mean_flux_samp, L_rescale_samp, dv_z_samp, coeffs_samp].
        @return: mean_trans: ndarray of shape (nsamp, nspec)
            Mean transmission curve for each sampled theta.
        '''

        _theta = np.atleast_2d(theta)
        nsamp = len(theta)
        mean_trans = np.ones((nsamp, self.nspec))

        for isamp, theta_now in enumerate(_theta):
            iF = find_closest(self.mean_flux_vec, theta_now[0])
            iL = find_closest(self.L_rescale_vec, theta_now[1])
            mean_trans[isamp, self.ipix_blu] = self.mean_t_prox[iF, iL]

        return mean_trans


    def simulateSpectra(self, nsamp=25000, stochastic=False):
        '''
        Simulate noiseless continua and absorption spectra. The absorption spectra contain the Ly-alpha forest and the
        proximity effect (approximately). The spectra are normalised to one at 1280 \AA..

        @param nsamp: int
        @param stochastic: bool
        @return:
            cont_norm: ndarray of shape (nsamp, nspec)
            flux_norm: ndarray of shape (nsamp, nspec)
            theta: ndarray of shape (nsamp, npca + 4)
        '''

        # sample theta
        theta = self.sample_theta(nsamp)

        if not stochastic:
            # manually set the mean flux values to the single value we want
            theta[:,0] = self.true_mean_flux

        # simulate the noiseless continua and absorption spectra
        cont, flux = self.simulator(theta, replace=(nsamp > self.nskew), ivar=None)

        # normalise the spectra to one at 1280 \AA
        flux_norm, cont_norm = normalise_spectra(self.wave_rest, flux, cont)

        return cont_norm, flux_norm, theta


    def assignNoise(self, half_dz, nsamp):
        '''
        Load empirical noise vectors from BOSS and assign random ones to the simulated spectra.
        Draw noise terms from the noise vectors.

        @param half_dz: float
        @param nsamp: int
        @return:
            ivar_vectors: ndarray of shape (nsamp, nspec)
            noise_terms: ndarray of shape (nsamp, nspec)
        '''

        zmin = self.z_qso - half_dz
        zmax = self.z_qso + half_dz

        # load the empirical noise vectors
        _, ivar_boss, gpm_boss = rebinNoiseVectors(zmin, zmax, self.wave_rest)

        # assign random noise vectors to the generated spectra
        rand_idx = np.random.randint(0, len(ivar_boss), size=nsamp)
        ivar_vectors = np.zeros((nsamp, self.nspec))
        for i in range(len(ivar_vectors)):
            ivar_vectors[i] = ivar_boss[rand_idx[i]]

        sigma_vectors = np.sqrt(1 / ivar_vectors)

        # draw noise terms from the noise vectors
        noise_terms = np.zeros((nsamp, self.nspec))
        rng = np.random.default_rng()
        for i in range(nsamp):
            noise_terms[i] = rng.normal(0, sigma_vectors[i], size=sigma_vectors.shape[-1])

        return ivar_vectors, noise_terms


class FullSimulator:
    def __init__(self, nsamp, z_qso, logLv_range, nlogL=10, npca=10, nskew=1000, wave_min=1000., wave_max=1970.,
                 fwhm=131.4, dloglam=1.0e-4, stochastic=False, half_dz=0.01, dvpix_red=500., train_frac=0.9):

        # initialise the ProximityWrapper
        self.Prox = ProximityWrapper(z_qso, logLv_range, nlogL, npca, nskew, wave_min, wave_max, fwhm, dloglam)

        # call the methods of ProximityWrapper and save everything to the object
        #self.mean_trans1d = self.Prox.meanTransmissionFromSkewers()
        #self.mean_trans = np.full((nsamp, self.Prox.nspec), self.mean_trans1d)
        self.cont, self.flux_noiseless, self.theta = self.Prox.simulateSpectra(nsamp, stochastic)
        self.mean_trans = self.Prox.meanTransmissioFromTheta(self.theta)
        self.ivar, noise_terms = self.Prox.assignNoise(half_dz, nsamp)
        self.flux = self.flux_noiseless + noise_terms

        self.wave_min = wave_min
        self.wave_max = wave_max
        self.nsamp = nsamp
        self.wave_rest = self.Prox.wave_rest
        self.dvpix_red = dvpix_red
        self.fwhm = self.Prox.fwhm
        self.dvpix = self.Prox.dvpix
        self.npca = self.Prox.npca
        self.nlogL = self.Prox.nlogL
        self.nskew = self.Prox.nskew
        self.L_mid = self.Prox.L_mid

        self._regrid(dvpix_red)
        #self.train_idcs, self.valid_idcs, self.test_idcs = self._split(train_frac)

        self.redshifts = np.full(self.nsamp, self.Prox.z_qso)
        #self.mags = np.full(self.nsamp, self.Prox.mag)

        # extract the L_rescale values from theta and use it to determine each QSOs logLv value
        _theta = np.atleast_2d(self.theta)
        L_rescale_samp = _theta[:,1]
        self.logLv_samp = np.log10(L_rescale_samp * self.L_mid)


    def _regrid(self, dvpix_red=500.):

        # construct the hybrid grid and the coarse grid
        self.wave_hybrid, _, _, _ = get_blu_red_wave_grid(self.wave_min, self.wave_max, self.Prox.wave_1216, self.dvpix,
                                                     dvpix_red)
        self.wave_coarse = get_wave_grid(self.wave_min, self.wave_max, dvpix_red)

        # interpolate/rebin everything
        interpolator_cont = interp1d(self.Prox.wave_rest, self.cont, kind="cubic", bounds_error=False,
                                     fill_value="extrapolate", axis=-1)
        self.cont_hybrid = interpolator_cont(self.wave_hybrid)
        self.cont_coarse = interpolator_cont(self.wave_coarse)

        self.flux_hybrid, ivar_hybrid, gpm_hybrid, _ = rebin_spectra(self.wave_hybrid, self.wave_rest, self.flux,
                                                                self.ivar, gpm=None)
        self.flux_coarse, ivar_coarse, gpm_coarse, _ = rebin_spectra(self.wave_coarse, self.wave_rest, self.flux,
                                                                self.ivar, gpm=None)

        self.ivar_hybrid = interpBadPixels(self.wave_hybrid, ivar_hybrid, gpm_hybrid)
        self.ivar_coarse = interpBadPixels(self.wave_coarse, ivar_coarse, gpm_coarse)

        #mean_trans_interpolator = interp1d(self.Prox.wave_rest, self.mean_trans1d, kind="cubic", bounds_error=False,
        #                                   fill_value="extrapolate")
        #mean_trans_hybrid1d = mean_trans_interpolator(self.wave_hybrid)
        #mean_trans_coarse1d = mean_trans_interpolator(self.wave_coarse)

        #self.mean_trans_hybrid = np.full((self.nsamp, len(self.wave_hybrid)), mean_trans_hybrid1d)
        #self.mean_trans_coarse = np.full((self.nsamp, len(self.wave_coarse)), mean_trans_coarse1d)

        mean_trans_interpolator = interp1d(self.Prox.wave_rest, self.mean_trans, kind="cubic", bounds_error=False,
                                           fill_value="extrapolate", axis=-1)
        self.mean_trans_hybrid = mean_trans_interpolator(self.wave_hybrid)
        self.mean_trans_coarse = mean_trans_interpolator(self.wave_coarse)


    def split(self, train_frac=0.9):

        valid_frac = 0.5 * (1 - train_frac)

        rng = np.random.default_rng()
        all_idcs = np.arange(0, self.nsamp)
        train_idcs = rng.choice(all_idcs, size=int(train_frac * self.nsamp), replace=False)
        valid_idcs = rng.choice(np.delete(all_idcs, train_idcs), size=int(valid_frac * self.nsamp), replace=False)
        test_idcs = np.delete(all_idcs, np.concatenate((train_idcs, valid_idcs)))

        return train_idcs, valid_idcs, test_idcs


    def saveFile(self, filepath="/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"):

        filename = "{}synthspec_BOSSlike_z{}_nsamp{}.hdf5".format(filepath, self.Prox.z_qso, self.nsamp)

        f = constructFile(self, filename)

        f.close()