import h5py
import os
import torch
from torch.utils.data import Dataset

def loadSynthFile(datapath=None, npca=10, z_qso=2.8, test=False):

    if datapath is None:
        datapath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"

    if test:
        size_descr = "small"
    else:
        size_descr = "large"

    filename = "{}synthspec_BOSSlike_npca{}_z{}_{}.hdf5".format(datapath, npca, z_qso, size_descr)

    f = h5py.File(filename, "r")

    return f, filename


class SynthSpectra(Dataset):
    '''
    Class for loading in synthetic spectra. Must be initialised for training, validation and testing separately.
    Automatically creates a 3D input array of flux, ivar and mean transmission on the hybrid grid.
    Returns the coarse-grid continua as "labels".
    '''

    def __init__(self, datapath=None, npca=10, z_qso=2.8, test=False, set="train"):
        '''

        @param datapath: str
            Path where the file is stored. If None, uses a default path.
        @param npca: int
            Number of PCA components. Default is 10.
        @param z_qso: float
            Redshift of the quasars. Default is 2.8.
        @param test: bool
            If True, loads in a smaller testing-only set. Default is False.
        @param set: str
            Must be one of ["train", "valid", "test"]. Indicates which set to load. Default is "train".
        '''

        if not ((set == "train") | (set == "valid") | (set == "test")):
            raise ValueError("Parameter 'set' must be either 'train', 'valid' or 'test'.")

        else:
            self.set = set
            self.grp_name = "/{}-data".format(set)

        self.file, self.filename = loadSynthFile(datapath, npca, z_qso, test)

        # load only the hybrid and coarse wavelength grid, the hybrid input and the hybrid true continua
        self.wave_fine = torch.FloatTensor(self.file["/meta/wave-fine"])
        self.wave_hybrid = torch.FloatTensor(self.file["/meta/wave-hybrid"])
        self.wave_coarse = torch.FloatTensor(self.file["/meta/wave-coarse"])

        self.flux_hybrid = torch.FloatTensor(self.file["{}/hybrid-grid/flux".format(self.grp_name)])
        self.ivar_hybrid = torch.FloatTensor(self.file["{}/hybrid-grid/ivar".format(self.grp_name)])
        self.mean_trans_hybrid = torch.FloatTensor(self.file["{}/hybrid-grid/mean-trans-flux".format(self.grp_name)])

        self.cont_hybrid = torch.FloatTensor(self.file["{}/hybrid-grid/cont".format(self.grp_name)])

        # TO DO: normalise the spectra to 1 at 1280 A in this class as well

        # make a 3D input tensor consisting of the absorption spectrum, the ivar and the transmission
        flux_unsq = torch.unsqueeze(self.flux_hybrid, dim=1)
        ivar_unsq = torch.unsqueeze(self.ivar_hybrid, dim=1)
        mean_trans_unsq = torch.unsqueeze(self.mean_trans_hybrid, dim=1)

        self.input_hybrid = torch.cat([flux_unsq, ivar_unsq, mean_trans_unsq], dim=1)

        self.n_qso = len(flux_unsq)

        # close the file again
        self.file.close()


    def __len__(self):
        return self.n_qso


    def __getitem__(self, idx):

        input_hybrid = self.input_hybrid[idx]
        cont = torch.unsqueeze(self.cont_hybrid, dim=1)[idx]

        return input_hybrid, cont


    @property
    def cont_coarse(self):

        f = h5py.File(self.filename, "r")
        cont_hybrid = torch.FloatTensor(f["{}/coarse-grid/cont".format(self.grp_name)])
        f.close()

        return cont_hybrid


    @property
    def cont_fine(self):

        f = h5py.File(self.filename, "r")
        cont_fine = torch.FloatTensor(f["{}/fine-grid/cont".format(self.grp_name)])
        f.close()

        return cont_fine


    @property
    def flux_fine(self):

        f = h5py.File(self.filename, "r")
        flux_fine = torch.FloatTensor(f["{}/fine-grid/flux".format(self.grp_name)])
        f.close()

        return flux_fine


    @property
    def flux_coarse(self):

        f = h5py.File(self.filename, "r")
        flux_coarse = torch.FloatTensor(f["{}/coarse-grid/flux".format(self.grp_name)])
        f.close()

        return flux_coarse


    @property
    def noise_fine(self):

        f = h5py.File(self.filename, "r")
        noise_fine = 1 / torch.sqrt(torch.FloatTensor(f["{}/fine-grid/ivar".format(self.grp_name)]))
        f.close()

        return noise_fine


    @property
    def noise_coarse(self):

        f = h5py.File(self.filename, "r")
        ivar_coarse = torch.FloatTensor(f["{}/coarse-grid/ivar".format(self.grp_name)])
        noise_coarse = 1 / torch.sqrt(ivar_coarse)
        f.close()

        return noise_coarse


    @property
    def noiseless_flux_fine(self):

        f = h5py.File(self.filename, "r")
        flux_noiseless = torch.FloatTensor(f["{}/fine-grid/noiseless-flux".format(self.grp_name)])
        f.close()

        return flux_noiseless