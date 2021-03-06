'''Module for loading in pytorch Datasets for the QSO spectra.'''

from torch.utils.data import Dataset, random_split
from data.load_data import load_synth_spectra, load_synth_noisy_cont, split_data, normalise_spectra, load_paris_spectra
from data.load_data import normalise_ivar
import numpy as np
#from pypeit.utils import fast_running_median
import torch

class Spectra(Dataset):
    def __init__(self, wave_grid, cont, flux, flux_smooth, norm1280=True,\
                 window=20, newnorm=False, ivar=None, normsmooth=False):
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wave_grid = wave_grid

        # if the ivar is already concatenated onto the flux, remove it
        # this prevents incorrect normalisation of the noise
        # particularly for splitting a set into train/validation/test
        if len(flux.shape) == 3:
            flux = flux[:,0,:]

        if norm1280:
            if newnorm:
                # normalise by dividing everything by the smoothed flux at 1280 A
                inwindow = (wave_grid > 1279) & (wave_grid < 1281)
                normfactor = flux_smooth[:,inwindow]
                cont = cont / normfactor
                flux = flux / normfactor
                flux_smooth_new = flux_smooth / normfactor

                if ivar is not None:
                    ivar = ivar ** normfactor**2

            else:
                if normsmooth:
                    # normalise such that the smoothed flux is 1 at 1280 (using a window)
                    flux_smooth_new, flux = normalise_spectra(wave_grid, flux_smooth, flux)
                    _, cont = normalise_spectra(wave_grid, flux_smooth, cont)

                    if ivar is not None:
                        _, ivar = normalise_ivar(wave_grid, flux_smooth, ivar)

                else:
                    # normalise such that the flux is 1 at 1280 (using a window)
                    flux_new, flux_smooth = normalise_spectra(wave_grid, flux, flux_smooth)
                    _, cont = normalise_spectra(wave_grid, flux, cont)

                    if ivar is not None:
                        _, ivar = normalise_ivar(wave_grid, flux, ivar)

                    flux = flux_new
                    flux_smooth_new = flux_smooth

        else:
            flux_smooth_new = flux_smooth

#        self.flux = torch.FloatTensor(flux).to(self.device)
#        self.cont = torch.FloatTensor(cont).to(self.device)
#        self.flux_smooth = torch.FloatTensor(flux_smooth_new).to(self.device)
        self.flux = torch.FloatTensor(flux)
        self.cont = torch.FloatTensor(cont)
        self.flux_smooth = torch.FloatTensor(flux_smooth_new)

        if ivar is not None:
            self.ivar = torch.FloatTensor(ivar)
        else:
            self.ivar = ivar

    def __len__(self):
        return len(self.flux)

    def __getitem__(self, idx):
        flux = self.flux[idx]
        flux_smooth = self.flux_smooth[idx]
        cont = self.cont[idx]

        return flux, flux_smooth, cont

    def add_channel_shape(self, n_channels=1):

        reshaped_specs = []
        for spec in [self.flux, self.flux_smooth, self.cont]:
            spec = spec.reshape((len(spec), n_channels, spec.shape[-1]))
            reshaped_specs.append(spec)

        self.flux = reshaped_specs[0]
        self.flux_smooth = reshaped_specs[1]
        self.cont = reshaped_specs[2]


    def add_noise_channel(self):
        '''
        Add a channel for the noise vectors of the spectra. Automatically calls add_channel_shape().
        @return:
        '''

        self.add_channel_shape(n_channels=1)

        if self.ivar is None:
            raise ValueError("No noise vectors provided.")

        else:
            # add the noise vector as a channel to the flux tensor
            # we don't attach it to the smoothed flux because it has become obsolete
            # we don't attach it to the continuum for compatibility with predictions
            ivar_reshaped = self.ivar.reshape((len(self.ivar), 1, self.ivar.shape[-1]))

            expanded_specs = []
            for spec in [self.flux]:
                expanded_specs.append(torch.cat((spec, ivar_reshaped), dim=1))

            self.flux = expanded_specs[0]

            print ("Attached an inverse variance channel to self.flux.")




class SynthSpectra(Spectra):
    '''Needs rewriting and new spectra for forest=True to be consistent.'''
    def __init__(self, regridded=True, small=False, npca=10,\
                       noise=True, norm1280=True, forest=True, window=20,\
                newnorm=False, homosced=True, poisson=False, SN=10,\
                 datapath=None, wave_split=None, boss=True, hetsced=True,
                 bossnoise=True, test=False):

        if not forest:
            wave_grid, cont, flux, flux_smooth = load_synth_noisy_cont(npca, smooth=True,\
                                                          window=window, homosced=homosced,\
                                                                       poisson=poisson, SN=SN,\
                                                                       datapath=datapath)

        else:
            if noise & (not hetsced):
                wave_grid, cont, flux, flux_smooth = load_synth_spectra(regridded,\
                                                                        small=False,\
                                                                        npca=npca,\
                                                                        noise=True,\
                                                                        datapath=datapath,\
                                                                        wave_split=wave_split,
                                                                        boss=boss)
            elif noise & hetsced:
                print ("Using noise & hetsced in SynthSpectra")
                wave_grid, cont, flux, flux_smooth, ivar = load_synth_spectra(regridded,
                                                                              small=False,
                                                                              npca=npca,
                                                                              noise=True,
                                                                              datapath=datapath,
                                                                              wave_split=wave_split,
                                                                              boss=boss,
                                                                              hetsced=hetsced,
                                                                              bossnoise=bossnoise,
                                                                              test=test)

            else:
                wave_grid, cont, flux = load_synth_spectra(regridded, small, npca,\
                                                           noise=False,\
                                                           datapath=datapath)

                # also smooth the spectra
                from pypeit.utils import fast_running_median
                flux_smooth = np.zeros(flux.shape)
                for i, F in enumerate(flux):
                    flux_smooth[i, :] = fast_running_median(F, window_size=window)

        if not hetsced:
            ivar = None

        super(SynthSpectra, self).__init__(wave_grid, cont, flux, flux_smooth,\
                                           norm1280, window=window, newnorm=newnorm,
                                           ivar=ivar)

    def split(self, usenoise=True):
        '''Needs to change to keep flux and flux_smooth together.
        Can use torch.utils.data.dataset.random_split()'''

        lengths = (np.array([0.9, 0.05, 0.05])*len(self)).astype(int)

        trainset, validset, testset = random_split(self, lengths)

        splitsets = []
        for el in [trainset, validset, testset]:

            if (self.ivar is not None) & usenoise:

                set = Spectra(self.wave_grid, self.cont[el.indices],
                              self.flux[el.indices], self.flux_smooth[el.indices],
                              norm1280=False, ivar=self.ivar[el.indices])
                set.add_noise_channel()
                splitsets.append(set)

            else:
                set = Spectra(self.wave_grid, self.cont[el.indices], self.flux[el.indices],
                              self.flux_smooth[el.indices], norm1280=False, ivar=None)
                set.add_channel_shape()
                splitsets.append(set)

        self.trainset, self.validset, self.testset = splitsets

        return self.trainset, self.validset, self.testset


class ParisContinua(Spectra):
    def __init__(self, noise=False, version=2, datapath=None):

        wave_grid, cont, flux, flux_smooth = load_paris_spectra(noise, version, datapath)

        super(ParisContinua, self).__init__(wave_grid, cont, flux, flux_smooth, \
                                           norm1280=True)

    def split(self):
        '''Needs to change to keep flux and flux_smooth together.
        Can use torch.utils.data.dataset.random_split()'''

        lengths = (np.array([0.9, 0.05, 0.05])*len(self)).astype(int)

        trainset, validset, testset = random_split(self, lengths)

        splitsets = []
        for el in [trainset, validset, testset]:
            splitsets.append(Spectra(self.wave_grid, self.cont[el.indices],\
                                     self.flux[el.indices], self.flux_smooth[el.indices],\
                                     norm1280=False))

        self.trainset, self.validset, self.testset = splitsets

        return self.trainset, self.validset, self.testset