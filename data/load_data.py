import numpy as np


def load_synth_spectra(regridded=True, small=False, npca=10,\
                       noise=False, SN=10, datapath=None,\
                       wave_split=None, boss=False, hetsced=False,
                       bossnoise=False, test=False):

    if datapath is None:
        datapath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"

    if bossnoise & regridded:
        if test:
            print ("Using test-only hybrid-grid spectra with BOSS noise.")
            filename = "{}forest_spectra_BOSSnoise_npca{}BOSS-regridded_test-only.npy".format(datapath, npca)
            data = np.load(filename)
        else:
            print ("Using bossnoise & regridded in load_synth_spectra")
            # this is the setting we'll most likely be using
            filename = "{}forest_spectra_BOSSnoise_npca{}BOSS-regridded.npy".format(datapath, npca)
            data = np.load(filename)

    elif bossnoise and not regridded:
        if test:
            print ("using test-only uniform grid spectra with BOSS noise.")
            filename = "{}forest_spectra_BOSSnoise_npca{}BOSS-grid_test-only.npy".format(datapath, npca)
        else:
            filename = "{}forest_spectra_BOSSnoise_npca{}BOSS-grid.npy".format(datapath, npca)

        data = np.load(filename)

    elif noise:
        if boss:
            if (not hetsced) & regridded:
                data = np.load(datapath + "forest_spectra_with_noiseSN"+str(SN)+"_npca"+str(npca)+"BOSS-regridded.npy")
            elif (not hetsced) & (not regridded):
                data = np.load(datapath + "forest_spectra_with_noiseSN"+str(SN)+"_npca"+str(npca)+"BOSS-grid.npy")
            elif hetsced & regridded:
                data = np.load(datapath + "forest_spectra_hetsced_noiseSN10-100_npca"+str(npca)+"BOSS-regridded.npy")
            elif hetsced & (not regridded):
                data = np.load(datapath + "forest_spectra_hetsced_noiseSN10-100_npca"+str(npca)+"BOSS-grid.npy")

        else:
            if regridded:
                if (wave_split is None) or (wave_split == 1216):
                    data = np.load(datapath + "forest_spectra_with_noiseSN"+str(SN)+"_regridded_npca" + str(npca) + "smooth-window20.npy")
                else:
                    data = np.load(datapath + "forest_spectra_with_noiseSN"+str(SN)+"_regridded_npca" + str(npca) + "smooth-window20_split"+str(int(wave_split))+".npy")

            else:
                data = np.load(datapath + "forest_spectra_with_noiseSN"+str(SN)+"_npca"+str(npca)+"smooth-window20.npy")

    elif npca==10:
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

    print ("Filename:", filename)

    if noise:
        if not hetsced:
            flux_smooth = data[:,:,3]
            return wave_grid, qso_cont, qso_flux, flux_smooth
        else:
            flux_smooth = data[:,:,3]
            ivar = data[:,:,4]
            return wave_grid, qso_cont, qso_flux, flux_smooth, ivar

    else:
        return wave_grid, qso_cont, qso_flux


def load_synth_noisy_cont(npca=10, smooth=False, window=20, homosced=True,\
                          poisson=False, SN=10, datapath=None):
    '''Convenience function for loading the synthetic continua with homoscedastic
    noise. qso_cont contains the continua, qso_flux contain the noisy continua.'''

    if datapath is None:
        datapath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
    npca_str = str(npca)

    if smooth:
        if homosced:
            data = np.load(datapath+"continua_with_noiseSN"+str(SN)+"_regridded_npca"+npca_str+"smooth-window"+str(window)+".npy")

        else:
            if poisson:
                if SN==10:
                    data = np.load(datapath+"continua_scaled-poisson-noise_regridded_npca"+npca_str+"smooth-window"+str(window)+".npy")
                else:
                    data = np.load(datapath+"continua_scaled-poisson-noiseSN"+str(SN)+"_regridded_npca"+npca_str+"smooth-window"+str(window)+".npy")
            else:
                data = np.load(datapath+"continua_with_constSNnoise_regridded_npca"+npca_str+"smooth-window"+str(window)+".npy")


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


def load_paris_spectra(noise=False, version=2, datapath=None):
    '''Convenience function for loading the Paris hand-fit continua with
    a simulated Ly-alpha forest and optional noise added in.'''

    if datapath is None:
        mainpath = "/net/vdesk/data2/buiten/MRP2/Data/"
    else:
        mainpath = datapath

    if noise:
        if version == 1:
            filename = mainpath + "paris_noisyflux_regridded.npy"
        else:
            filename = mainpath + "paris_noisyflux_regridded_v"+str(version)+".npy"
    else:
        filename = mainpath + "paris_noiselessflux_regridded.npy"

    data = np.load(filename)
    wave_grid = data[0,:,0]
    cont = data[:,:,1]
    flux = data[:,:,2]
    flux_smooth = data[:,:,3]

    return wave_grid, cont, flux, flux_smooth

def split_data(attributes, targets, train_size=0.9, test_size=0.05):
    from sklearn.model_selection import train_test_split
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


def normalise_ivar(wave_grid, flux, ivar, windowmin=1270, windowmax=1290):

    try:
        wave_grid1d = wave_grid[0,:]
    except:
        wave_grid1d = wave_grid

    window = (wave_grid1d > windowmin) & (wave_grid1d < windowmax)
    flux_median_window = np.median(flux[:,window], axis=1)
    flux_norm = np.zeros(flux.shape)
    ivar_norm = np.zeros(ivar.shape)
    for i in range(len(flux)):
        flux_norm[i,:] = flux[i,:] / flux_median_window[i]
        ivar_norm[i,:] = ivar[i,:] * flux_median_window[i]**2

    return flux_norm, ivar_norm