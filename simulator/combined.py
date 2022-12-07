'''Module for combining spectra generated from several simulators.'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
from simulator.singular import FullSimulator
from simulator.save import constructFile, constructTransmissionTemplates
from scipy.interpolate import interp1d
from IPython import embed


class SimulationFromFile:

    def __init__(self, filename):

        print ("Loading simulation from file {}.".format(filename))

        f = h5py.File(filename, "r")

        self.fwhm = f["meta"].attrs["fwhm"]
        self.dvpix = f["meta"].attrs["dv-fine"]
        self.dvpix_red = f["meta"].attrs["dv-coarse"]
        self.npca = f["meta"].attrs["npca"]
        self.nskew = f["meta"].attrs["nskew"]
        self.wave_split = f["meta"].attrs["wave-split"]

        self.wave_rest = np.copy(f["meta"]["wave-fine"])
        self.wave_hybrid = np.copy(f["meta"]["wave-hybrid"])

        self.mean_trans = np.copy(f["fine"]["mean-trans-flux"])
        self.cont = np.copy(f["fine"]["cont"])
        self.ivar = np.copy(f["fine"]["ivar"])
        self.flux = np.copy(f["fine"]["flux"])
        self.flux_noiseless = np.copy(f["fine"]["flux-noiseless"])

        self.redshifts = np.copy(f["meta"]["redshifts"])
        self.logLv_samp = np.copy(f["meta"]["logLv"])

        self.cont_hybrid = np.copy(f["hybrid"]["cont"])
        self.flux_hybrid = np.copy(f["hybrid"]["flux"])
        self.ivar_hybrid = np.copy(f["hybrid"]["ivar"])
        self.mean_trans_hybrid = np.copy(f["hybrid"]["mean-trans-flux"])

        # extract the transmission templates
        self.z_mid = np.copy(f["meta"]["z-mid"])
        self.logLv_samp = np.copy(f["meta"]["logLv"])
        self.logLv_mid = np.copy(f["meta"]["logLv-mid"])
        self.mean_t_prox0 = np.copy(f["meta"]["trans-templates"])

        f.close()


class CombinedSimulations:
    '''
    Class for combining the spectra generated from multiple simulators.
    '''

    # TODO: remove coarse-grid references
    # currently the FullSimulator sets coarse-grid attributes except for the wavelength grid to None

    # TODO: load the data from files instead of a simulator object

    def __init__(self, sims_list):

        assert isinstance(sims_list, list)

        if isinstance(sims_list[0], str):
            # overwrite sims_list with the loaded simulations
            sims_list = self._load_from_files(sims_list)
            fromfile = True

        else:
            assert isinstance(sims_list[0], FullSimulator)
            fromfile = False

        # for metadata, use the values of the first simulator object
        self.fwhm = sims_list[0].fwhm
        self.dvpix = sims_list[0].dvpix
        self.dvpix_red = sims_list[0].dvpix_red
        self.npca = sims_list[0].npca
        self.nskew = sims_list[0].nskew
        self.wave_split = sims_list[0].wave_split

        self.wave_rest = sims_list[0].wave_rest
        #self.wave_coarse = sims_list[0].wave_coarse
        self.wave_coarse = None
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

        # coarse-grid attributes are None so they can't be concatenated
        self.cont_hybrid = np.concatenate([sim.cont_hybrid for sim in sims_list], axis=0)
        #self.cont_coarse = np.concatenate([sim.cont_coarse for sim in sims_list], axis=0)
        self.flux_hybrid = np.concatenate([sim.flux_hybrid for sim in sims_list], axis=0)
        #self.flux_coarse = np.concatenate([sim.flux_coarse for sim in sims_list], axis=0)
        self.ivar_hybrid = np.concatenate([sim.ivar_hybrid for sim in sims_list], axis=0)
        #self.ivar_coarse = np.concatenate([sim.ivar_coarse for sim in sims_list], axis=0)
        self.mean_trans_hybrid = np.concatenate([sim.mean_trans_hybrid for sim in sims_list], axis=0)
        #self.mean_trans_coarse = np.concatenate([sim.mean_trans_coarse for sim in sims_list], axis=0)

        # derive the number of samples in the combined set
        self.nsamp = len(self.cont)
        self.nsets = len(sims_list)

        # extract the mean transmission templates and 3D grid midpoints (z, logLv, wav)
        self.trans_templates = np.array([sim.mean_t_prox0 for sim in sims_list])

        if fromfile:
            self.z_mids = np.array([sim.z_mid for sim in sims_list])
            self.logLv_mids = np.array([sim.logLv_mid for sim in sims_list])
        else:
            self.z_mids = np.array([sim.Prox.z_qso for sim in sims_list])
            self.logLv_mids = np.log10(np.array(sims_list[0].Prox.L_rescale_vec * sims_list[0].Prox.L_mid))
        #self.logLv_mids = np.log10( np.array( [sim.Prox.L_rescale_vec * sim.Prox.L_mid for sim in sims_list] ) )

        # take only the first sims_list entry for the logLv axis on the grid, since we fixed the logLv range
        # TODO: edit this because we're no longer fixing the logLv range
        #self.logLv_mids = np.array([sim.logLv_mid for sim in sims_list])
        #self.logLv_mids = np.log10( np.array( sims_list[0].Prox.L_rescale_vec * sims_list[0].Prox.L_mid ) )
        # wavelength grids are already extracted above

        # interpolate the transmission templates onto the hybrid grid as well
        interpolator = interp1d(self.wave_rest, self.trans_templates, kind="cubic", bounds_error=False,
                                fill_value="extrapolate", axis=-1)
        self.trans_templates_hybrid = interpolator(self.wave_hybrid)



    def split(self, train_frac=0.9):

        valid_frac = 0.5 * (1 - train_frac)

        rng = np.random.default_rng()
        all_idcs = np.arange(0, self.nsamp)
        train_idcs = rng.choice(all_idcs, size=int(train_frac * self.nsamp), replace=False)
        valid_idcs = rng.choice(np.delete(all_idcs, train_idcs), size=int(valid_frac * self.nsamp), replace=False)
        test_idcs = np.delete(all_idcs, np.concatenate((train_idcs, valid_idcs)))

        return train_idcs, valid_idcs, test_idcs


    def _load_from_files(self, file_list):
        '''Load the data from the files in file_list.'''

        sims_list = []

        for f in file_list:
            sims_list.append(SimulationFromFile(f))

        return sims_list


    def saveFile(self, filepath="/net/vdesk/data2/buiten/MRP2/pca-sdss-old/", dz=None, train_frac=0.9):

        if dz is None:
            filename = "{}synthspec_combined_{}sets.hdf5".format(filepath, self.nsets)
        else:
            if self.nsamp == 25000:
                filename = "{}synthspec_combined_dz{}.hdf5".format(filepath, dz)
            else:
                filename = "{}synthspec_combined_dz{}_nsamp{}.hdf5".format(filepath, dz, self.nsamp)

        f = constructFile(self, filename, train_frac)

        f.close()


    def saveTransmissionTemplates(self, filepath="/net/vdesk/data2/buiten/MRP2/Data/", dz=None):

        if dz is None:
            filename = "{}transmission_templates_{}sets.hdf5".format(filepath, self.nsets)
        else:
            filename = "{}transmission_templates_dz{}.hdf5".format(filepath, dz)

        f = constructTransmissionTemplates(self, filename)

        f.close()


    def plotExample(self):
        '''
        Plot the continuum, absorption spectrum and mean transmission profile for a single random example.

        @return:
        '''

        rand_idx = np.random.randint(0, self.nsamp)

        fig, ax = plt.subplots()

        ax.plot(self.wave_rest, self.cont[rand_idx], lw=1.5, c="tab:orange", alpha=.7, label="Continuum")
        ax.plot(self.wave_rest, self.flux[rand_idx], lw=1., c="tab:blue", alpha=.7, label="Absorption spectrum")
        ax.plot(self.wave_rest, self.mean_trans[rand_idx], lw=1., c="tab:red", alpha=.6, label="Mean transmission")

        ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"$F / F_{1280 \AA}$")

        ax.legend()
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)

        plt.show()