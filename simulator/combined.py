'''Module for combining spectra generated from several simulators.'''

import numpy as np
from simulator.singular import FullSimulator
from simulator.save import constructFile

class CombinedSimulations:
    '''
    Class for combining the spectra generated from multiple simulators.
    '''

    def __init__(self, sims_list):

        assert isinstance(sims_list, list)
        assert isinstance(sims_list[0], FullSimulator)

        # for metadata, use the values of the first simulator object
        self.fwhm = sims_list[0].fwhm
        self.dvpix = sims_list[0].dvpix
        self.dvpix_red = sims_list[0].dvpix_red
        self.npca = sims_list[0].npca
        self.nskew = sims_list[0].nskew
        self.wave_split = sims_list[0].wave_split

        self.wave_rest = sims_list[0].wave_rest
        self.wave_coarse = sims_list[0].wave_coarse
        self.wave_hybrid = sims_list[0].wave_hybrid

        # concatenate the data from all simulators
        # we want to be able to use instances of this class as the simulator argument in constructFile
        self.mean_trans = np.concatenate([sim.mean_trans for sim in sims_list], axis=0)
        self.cont = np.concatenate([sim.cont for sim in sims_list], axis=0)
        self.ivar = np.concatenate([sim.ivar for sim in sims_list], axis=0)
        self.flux = np.concatenate([sim.flux for sim in sims_list], axis=0)
        self.flux_noiseless = np.concatenate([sim.flux_noiseless for sim in sims_list], axis=0)

        self.redshifts = np.concatenate([sim.redshifts for sim in sims_list], axis=0)
        #self.mags = np.concatenate([sim.mags for sim in sims_list], axis=0)
        self.logLv_samp = np.concatenate([sim.logLv_samp for sim in sims_list], axis=0)

        self.cont_hybrid = np.concatenate([sim.cont_hybrid for sim in sims_list], axis=0)
        self.cont_coarse = np.concatenate([sim.cont_coarse for sim in sims_list], axis=0)
        self.flux_hybrid = np.concatenate([sim.flux_hybrid for sim in sims_list], axis=0)
        self.flux_coarse = np.concatenate([sim.flux_coarse for sim in sims_list], axis=0)
        self.ivar_hybrid = np.concatenate([sim.ivar_hybrid for sim in sims_list], axis=0)
        self.ivar_coarse = np.concatenate([sim.ivar_coarse for sim in sims_list], axis=0)
        self.mean_trans_hybrid = np.concatenate([sim.mean_trans_hybrid for sim in sims_list], axis=0)
        self.mean_trans_coarse = np.concatenate([sim.mean_trans_coarse for sim in sims_list], axis=0)

        # derive the number of samples in the combined set
        self.nsamp = len(self.cont)
        self.nsets = len(sims_list)


    def split(self, train_frac=0.9):

        valid_frac = 0.5 * (1 - train_frac)

        rng = np.random.default_rng()
        all_idcs = np.arange(0, self.nsamp)
        train_idcs = rng.choice(all_idcs, size=int(train_frac * self.nsamp), replace=False)
        valid_idcs = rng.choice(np.delete(all_idcs, train_idcs), size=int(valid_frac * self.nsamp), replace=False)
        test_idcs = np.delete(all_idcs, np.concatenate((train_idcs, valid_idcs)))

        return train_idcs, valid_idcs, test_idcs


    def saveFile(self, filepath="/net/vdesk/data2/buiten/MRP2/pca-sdss-old/", dz=None):

        if dz is None:
            filename = "{}synthspec_combined_{}sets.hdf5".format(filepath, self.nsets)
        else:
            filename = "{}synthspec_combined_dz{}.hdf5".format(filepath, dz)

        f = constructFile(self, filename)

        f.close()