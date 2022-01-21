import numpy as np
from sklearn.model_selection import train_test_split


def load_synth_spectra(regridded=True, small=False, npca=10,\
                       noise=False):
    datapath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
    if npca==10:
        if noise:
            data = np.load(datapath+"forest_spectra_with_noise_regridded.npy")
        else:
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
        if noise:
            data = np.load(datapath+"forest_spectra_with_noise_regridded_npca"+str(npca)+".npy")
        else:
            if regridded:
                data = np.load(datapath+"gen_spectrum_regridded_big_array_npca"+str(npca)+".npy")
            else:
                data = np.load(datapath+"gen_spectrum_nonregridded_big_array_npca"+str(npca)+".npy")

    wave_grid = data[0,:,0]
    qso_cont = data[:,:,1]
    qso_flux = data[:,:,2]

    return wave_grid, qso_cont, qso_flux


def load_synth_noisy_cont(npca=10, smooth=False, window=20):
    '''Convenience function for loading the synthetic continua with homoscedastic
    noise. qso_cont contains the continua, qso_flux contain the noisy continua.'''

    datapath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
    npca_str = str(npca)

    if smooth:
        data = np.load(datapath+"continua_with_noise_regridded_npca"+npca_str+"smooth-window"+str(window)+".npy")

    else:
        data = np.load(datapath+"continua_with_noise_regridded_npca"+npca_str+".npy")

    wave_grid = data[0,:,0]
    qso_cont = data[:,:,1]
    qso_flux = data[:,:,2]

    if smooth:
        qso_flux_smooth = data[:,:,3]
        return wave_grid, qso_cont, qso_flux, qso_flux_smooth

    else:
        return wave_grid, qso_cont, qso_flux


def load_paris_spectra(noise=False):
    '''Convenience function for loading the Paris hand-fit continua with
    a simulated Ly-alpha forest and optional noise added in.'''

    mainpath = "/net/vdesk/data2/buiten/MRP2/Data/"

    if noise:
        filename = mainpath + "paris_noisyflux_regridded.npy"
    else:
        filename = mainpath + "paris_noiselessflux_regridded.npy"

    data = np.load(filename)
    wave_grid = data[0,:,0]
    cont = data[:,:,1]
    flux = data[:,:,2]

    return wave_grid, cont, flux

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


def normalise_spectra(wave_grid, flux, cont, windowmin=1270, windowmax=1290):

    try:
        wave_grid1d = wave_grid[0,:]
    except:
        wave_grid1d = wave_grid

    window = (wave_grid1d > windowmin) & (wave_grid1d < windowmax)
    flux_median_window = np.median(flux[:,window], axis=1)
    flux_norm = np.zeros(flux.shape)
    cont_norm = np.zeros(cont.shape)
    for i in range(len(flux)):
        flux_norm[i,:] = flux[i,:]/flux_median_window[i]
        cont_norm[i,:] = cont[i,:]/flux_median_window[i]

    return flux_norm, cont_norm