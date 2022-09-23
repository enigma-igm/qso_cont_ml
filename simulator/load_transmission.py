import h5py
import torch
from torch import FloatTensor

class TransmissionTemplates:
    '''
    Class for loading a pre-created transmission template bank from disk.

    Attributes:
        wave_fine:
        wave_hybrid:
        mean_trans_fine:
        mean_trans_hybrid:
        z_mids:
        logLv_mids:

    TODO: write plotting method for visualisation?
    '''

    def __init__(self, filepath, dz):

        assert isinstance(filepath, str)
        assert isinstance(dz, float)

        filename = "{}transmission_templates_dz{}.hdf5".format(filepath, dz)

        f = h5py.File(filename, "r")

        self.wave_fine = FloatTensor(f["/fine-grid/wave-fine"])
        self.wave_hybrid = FloatTensor(f["/hybrid-grid/wave-hybrid"])

        self.mean_trans_fine = FloatTensor(f["/fine-grid/mean-trans"])
        self.mean_trans_hybrid = FloatTensor(f["/hybrid-grid/mean-trans"])

        self.z_mids = FloatTensor(f["z-mids"])
        self.logLv_mids = FloatTensor(f["logLv-mids"])

        f.close()