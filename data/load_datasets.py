'''Module for loading in pytorch Datasets for the QSO spectra.'''

from torch.utils.data import Dataset
from data.load_data import load_synth_spectra, split_data, normalise_spectra
import numpy as np
from pypeit.utils import fast_running_median

class Spectra(Dataset):
    def __init__(self, wave_grid, cont, flux, norm1280=True):
        self.wave_grid = wave_grid

        # also smooth the spectra
        flux_smooth = np.zeros(flux.shape)
        for i, F in enumerate(flux):
            flux_smooth[i,:] = fast_running_median(F, window_size=20)

        if norm1280:
            flux_smooth, flux = normalise_spectra(wave_grid, flux_smooth, flux)
            _, cont = normalise_spectra(wave_grid, flux_smooth, cont)

        self.flux = flux
        self.cont = cont
        self.flux_smooth = flux_smooth

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        flux = self.flux[idx]
        flux_smooth = self.flux_smooth[idx]
        cont = self.cont[idx]

        return flux, flux_smooth, cont


class SynthSpectra(Spectra):
    def __init__(self, regridded=True, small=False, npca=10,\
                       noise=False, norm1280=True):

        wave_grid, cont, flux = load_synth_spectra(regridded, small, npca,\
                                                   noise)

        super(SynthSpectra, self).__init__(wave_grid, cont, flux, norm1280)


    def split(self):

        flux_train, flux_valid, flux_test, cont_train, cont_valid, cont_test = split_data(self.flux, self.cont)
        self.trainset = Spectra(self.wave_grid, cont_train, flux_train, norm1280=False)
        self.validset = Spectra(self.wave_grid, cont_valid, flux_valid, norm1280=False)
        self.testset = Spectra(self.wave_grid, cont_test, flux_test, norm1280=False)

        return self.trainset, self.validset, self.testset