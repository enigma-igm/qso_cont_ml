import numpy as np
import torch
from data.load_datasets import SynthSpectra

class AutofitterSpectra(SynthSpectra):
    def __init__(self, redshifts, ivar=None, npca=10, SN=10, datapath=None):
        '''Needs clean splitter method.'''

        super().__init__(regridded=False, npca=npca, noise=True, SN=SN, datapath=datapath, boss=True)

        self.n_qso = self.flux.shape[0]
        self.n_pix = self.flux.shape[-1]

        if isinstance(redshifts, np.ndarray) and (redshifts.shape == (self.n_qso,)):
            self.redshifts = redshifts
        else:
            raise ValueError("Parameter 'redshifts' should be a 1D array of length n_qso.")

        self.wave_rest = np.broadcast_to(self.wave_grid, (self.n_qso, self.n_pix))
        self.wave_obs = self.wave_rest * (1 + redshifts)

        if ivar is None:
            # assume homoscedastic noise described by the given signal-to-noise ratio
            self.ivar = np.broadcast_to(SN**2, (self.n_qso, self.n_pix))
        else:
            self.ivar = ivar

