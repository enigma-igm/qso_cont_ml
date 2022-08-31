'''Module for modelling the uncertainty in redshift estimates with the purpose of smoothing mean transmission curves.'''

import numpy as np
import astropy.constants as const


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
    @param true_redshifts: ndarray of shape (n_qso,)
    @param dv_sigma: float
    @return: wave_rest_new: ndarray of shape (n_qso, n_spec)
    '''

    _wave_rest = np.atleast_2d(wave_rest)

    measured_redshifts = perturbRedshifts(true_redshifts, dv_sigma)
    wave_obs = _wave_rest * (1 + true_redshifts)
    wave_rest_new = wave_obs / (1 + measured_redshifts)

    return wave_rest_new