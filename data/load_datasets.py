'''Module for loading in pytorch Datasets for the QSO spectra.'''

from torch.utils.data import Dataset, random_split
from data.load_data import load_synth_spectra, load_synth_noisy_cont, split_data, normalise_spectra
import numpy as np
from pypeit.utils import fast_running_median

class Spectra(Dataset):
    def __init__(self, wave_grid, cont, flux, flux_smooth, norm1280=True,\
                 window=20, newnorm=False):
        self.wave_grid = wave_grid

        if norm1280:
            if newnorm:
                # normalise by dividing everything by the smoothed flux at 1280 A
                inwindow = (wave_grid > 1279) & (wave_grid < 1281)
                normfactor = flux_smooth[:,inwindow]
                cont = cont / normfactor
                flux = flux / normfactor
                flux_smooth_new = flux_smooth / normfactor

            else:

                flux_smooth_new, flux = normalise_spectra(wave_grid, flux_smooth, flux)
                _, cont = normalise_spectra(wave_grid, flux_smooth, cont)

        else:
            flux_smooth_new = flux_smooth

        self.flux = flux
        self.cont = cont
        self.flux_smooth = flux_smooth_new

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        flux = self.flux[idx]
        flux_smooth = self.flux_smooth[idx]
        cont = self.cont[idx]

        return flux, flux_smooth, cont


class SynthSpectra(Spectra):
    '''Needs rewriting and new spectra for forest=True to be consistent.'''
    def __init__(self, regridded=True, small=False, npca=10,\
                       noise=False, norm1280=True, forest=True, window=20,\
                newnorm=False):

        if not forest:
            wave_grid, cont, flux, flux_smooth = load_synth_noisy_cont(npca, smooth=True,\
                                                          window=window)

        else:
            wave_grid, cont, flux = load_synth_spectra(regridded, small, npca,\
                                                       noise)

            # also smooth the spectra
            flux_smooth = np.zeros(flux.shape)
            for i, F in enumerate(flux):
                flux_smooth[i, :] = fast_running_median(F, window_size=window)

        super(SynthSpectra, self).__init__(wave_grid, cont, flux, flux_smooth,\
                                           norm1280, window=window, newnorm=newnorm)

    def split(self):
        '''Needs to change to keep flux and flux_smooth together.
        Can use torch.utils.data.dataset.random_split()'''

        lengths = (np.array([0.9, 0.05, 0.05])*len(self)).astype(int)

        trainset, validset, testset = random_split(self, lengths)
        #train_idx = trainset.indices
        #valid_idx = validset.indices
        #test_idx = testset.indices

        splitsets = []
        for el in [trainset, validset, testset]:
            splitsets.append(Spectra(self.wave_grid, self.cont[el.indices],\
                                     self.flux[el.indices], self.flux_smooth[el.indices],\
                                     norm1280=False))

        self.trainset, self.validset, self.testset = splitsets


        #flux_train, flux_valid, flux_test, cont_train, cont_valid, cont_test = split_data(self.flux, self.cont)
        #self.trainset = Spectra(self.wave_grid, cont_train, flux_train, norm1280=False)
        #self.validset = Spectra(self.wave_grid, cont_valid, flux_valid, norm1280=False)
        #self.testset = Spectra(self.wave_grid, cont_test, flux_test, norm1280=False)

        return self.trainset, self.validset, self.testset