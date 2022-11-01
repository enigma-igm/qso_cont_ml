'''Useful functions for normalisation.'''

import numpy as np

def normalise_spectra(wave_grid, flux, cont, windowmin=1270, windowmax=1290):

    try:
        wave_grid1d = wave_grid[0,:]
    except:
        wave_grid1d = wave_grid

    window = (wave_grid1d > windowmin) & (wave_grid1d < windowmax)
    flux_median_window = np.median(flux[:,window], axis=1)
    flux_norm = np.zeros(flux.shape)
    cont_norm = np.zeros(cont.shape)
    for i in range(len(flux)):
        flux_norm[i,:] = flux[i,:]/flux_median_window[i]
        cont_norm[i,:] = cont[i,:]/flux_median_window[i]

    return flux_norm, cont_norm


def normalise_ivar(wave_grid, flux, ivar, windowmin=1270, windowmax=1290):

    try:
        wave_grid1d = wave_grid[0,:]
    except:
        wave_grid1d = wave_grid

    window = (wave_grid1d > windowmin) & (wave_grid1d < windowmax)
    flux_median_window = np.median(flux[:,window], axis=1)
    flux_norm = np.zeros(flux.shape)
    ivar_norm = np.zeros(ivar.shape)
    for i in range(len(flux)):
        flux_norm[i,:] = flux[i,:] / flux_median_window[i]
        ivar_norm[i,:] = ivar[i,:] * flux_median_window[i]**2

    return flux_norm, ivar_norm