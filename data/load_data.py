import numpy as np
from sklearn.model_selection import train_test_split

def load_synth_spectra(regridded=True, small=False, npca=10):
    datapath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
    if npca==10:
        if regridded:
            if small:
                data = np.load(datapath+"gen_spectrum_regridded_array.npy")
            else:
                data = np.load(datapath+"gen_spectrum_regridded_big_array.npy")
        else:
            if small:
                data = np.load(datapath+"gen_spectrum_nonregridded_array.npy")
            else:
                data = np.load(datapath+"gen_spectrum_nonregridded_big_array.npy")

    else:
        if regridded:
            data = np.load(datapath+"gen_spectrum_regridded_big_array_npca"+str(npca)+".npy")
        else:
            data = np.load(datapath+"gen_spectrum_nonregridded_big_array_npca"+str(npca)+".npy")

    wave_grid = data[0,:,0]
    qso_cont = data[:,:,1]
    qso_flux = data[:,:,2]

    return wave_grid, qso_cont, qso_flux


def split_data(attributes, targets, train_size=0.9, test_size=0.05):
    rest_size = 1 - train_size
    X_train, X_rest, y_train, y_rest = train_test_split(attributes, targets,\
                                                        test_size=rest_size,\
                                                        random_state=0)

    test_size_of_rest = test_size/rest_size
    X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest,\
                                                        test_size=test_size_of_rest,
                                                        random_state=0)

    return X_train, X_valid, X_test, y_train, y_valid, y_test