'''Module for modelling the uncertainty in redshift estimates with the purpose of smoothing mean transmission curves.'''

import numpy as np
import astropy.constants as const
from scipy.ndimage import gaussian_filter1d


def velShiftToRedshiftPerturbation(dv):
    '''
    Converts a velocity perturbation dv to the corresponding redshift perturbation dz, using dv = c * dz.

    @param dv: float
    @return: dz: float
    '''

    c_light = const.c.to("km/s").value
    dz = dv / c_light

    return dz


def perturbRedshifts(redshifts, dv_sigma=700):
    '''
    Perturb a set of redshifts by Gaussian noise with standard deviation dv_sigma in velocity space.

    @param redshifts: ndarray of shape (n_qso,)
    @param dv_sigma: float
    @return: measured_redshifts: ndarray of shape (n_qso,)
    '''

    dz_sigma = velShiftToRedshiftPerturbation(dv_sigma)

    # draw redshift perturbation terms from a gaussian with sigma = dv_sigma
    rng = np.random.default_rng()
    dzs = rng.normal(scale=dz_sigma, size=len(redshifts))
    measured_redshifts = redshifts + dzs

    return measured_redshifts


def modelRedshiftUncertainty(wave_rest, true_redshifts, dv_sigma=700):
    '''
    Incorporate modelled redshift uncertainty in the rest-frame wavelengths of a set of spectra.

    @param wave_rest: ndarray of shape (n_qso, n_spec) or (n_spec,)
    @param true_redshifts: ndarray of shape (n_qso,) or float
    @param dv_sigma: float
    @return: wave_rest_new: ndarray of shape (n_qso, n_spec)
    @return: measured redshifts: ndarray of shape (n_qso,)
    '''

    _wave_rest = np.atleast_2d(wave_rest)
    _true_redshifts = np.atleast_1d(true_redshifts)

    measured_redshifts = perturbRedshifts(_true_redshifts, dv_sigma)
    wave_rest_new = np.zeros((len(_true_redshifts), _wave_rest.shape[-1]))

    for i in range(len(_true_redshifts)):
        wave_obs = _wave_rest * (1 + _true_redshifts[i])
        wave_rest_new[i] = wave_obs / (1 + measured_redshifts[i])

    return wave_rest_new, measured_redshifts


def smoothTransmission(wave_rest, mean_trans, dv_sigma=700):
    '''

    @param wave_rest:
    @param mean_trans: ndarray of shape (nsamp, nspec)
    @param dv_sigma: float
        Standard deviation in velocity space (km/s) for the Gaussian kernel used for smoothing.
    @return:
    '''

    c_light = const.c.to("km/s").value

    #loglam = np.log10(wave_rest)
    #dloglam = loglam[1:] - loglam[:-1]

    dloglam_sigma = dv_sigma / (c_light * np.log(10))

    mean_trans_smoothed = gaussian_filter1d(mean_trans, dloglam_sigma, axis=-1)

    return mean_trans_smoothed

