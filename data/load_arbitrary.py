import numpy as np
import torch
from torch.utils.data import Dataset
from simulator.load_transmission import TransmissionTemplates

class InputSpectra(Dataset):
    '''
    Class for storing and structuring the spectra for which we wish to get continuum predictions.
    The spectra are regridded onto the hybrid rest-frame wavelength grid of the network.
    An input tensor is created, containing the spectra, corresponding noise vectors (in terms of inverse variance) and
    the mean transmission curves corresponding to the redshift and Lyman-limit luminosity of each quasar.
    These mean transmission curves are determined through bilinear interpolation in (z, logLv) space based on the
    output of the simulator.
    '''

    def __init__(self, wave_grid, flux, ivar, redshifts, logLv, restframe=True, wave_min=980., wave_max=1970., dloglam=1e-4,
                 cont=None, wave_split=1260.):

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
        '''