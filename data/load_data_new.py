import h5py
import os

def loadSynthFile(npca=10, datapath=None, test=False, z_qso=2.8):

    if datapath is None:
        datapath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"

    if test:
        size_descr = "small"
    else:
        size_descr = "large"

    filename = "{}synthspec_BOSSlike_npca{}_z{}_{}.hdf5".format(datapath, npca, z_qso, size_descr)

    f = h5py.File(filename, "r")

    return f, filename


