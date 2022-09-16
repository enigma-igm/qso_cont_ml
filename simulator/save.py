import h5py

def constructFile(simulator, filename, train_frac=0.9):

    # split the data into training, validation and test data
    train_idcs, valid_idcs, test_idcs = simulator.split(train_frac)

    f = h5py.File(filename, "w")

    # create groups for training/validation/test data and for metadata
    grp_traindata = f.create_group("train-data")
    grp_validdata = f.create_group("valid-data")
    grp_testdata = f.create_group("test-data")
    grp_meta = f.create_group("meta")

    # create subgroups for the fine, coarse and hybrid grid
    grp_fine_train = grp_traindata.create_group("fine-grid")
    grp_coarse_train = grp_traindata.create_group("coarse-grid")
    grp_hybrid_train = grp_traindata.create_group("hybrid-grid")

    grp_fine_valid = grp_validdata.create_group("fine-grid")
    grp_coarse_valid = grp_validdata.create_group("coarse-grid")
    grp_hybrid_valid = grp_validdata.create_group("hybrid-grid")

    grp_fine_test = grp_testdata.create_group("fine-grid")
    grp_coarse_test = grp_testdata.create_group("coarse-grid")
    grp_hybrid_test = grp_testdata.create_group("hybrid-grid")

    grp_meta.create_dataset("wave-fine", data=simulator.wave_rest)
    grp_meta.create_dataset("wave-coarse", data=simulator.wave_coarse)
    grp_meta.create_dataset("wave-hybrid", data=simulator.wave_hybrid)

    # add the indexed spectra to the right group
    grps = [[grp_fine_train, grp_coarse_train, grp_hybrid_train],
            [grp_fine_valid, grp_coarse_valid, grp_hybrid_valid],
            [grp_fine_test, grp_coarse_test, grp_hybrid_test]]

    for (idcs, [grp_fine, grp_coarse, grp_hybrid]) in zip([train_idcs, valid_idcs, test_idcs], grps):

        grp_fine.create_dataset("cont", data=simulator.cont[idcs])
        grp_fine.create_dataset("flux", data=simulator.flux[idcs])
        grp_fine.create_dataset("ivar", data=simulator.ivar[idcs])
        grp_fine.create_dataset("mean-trans-flux", data=simulator.mean_trans[idcs])
        # also save the noiseless absorption spectrum
        grp_fine.create_dataset("noiseless-flux", data=simulator.flux_noiseless[idcs])

        grp_coarse.create_dataset("cont", data=simulator.cont_coarse[idcs])
        grp_coarse.create_dataset("flux", data=simulator.flux_coarse[idcs])
        grp_coarse.create_dataset("ivar", data=simulator.ivar_coarse[idcs])
        grp_coarse.create_dataset("mean-trans-flux", data=simulator.mean_trans_coarse[idcs])

        grp_hybrid.create_dataset("cont", data=simulator.cont_hybrid[idcs])
        grp_hybrid.create_dataset("flux", data=simulator.flux_hybrid[idcs])
        grp_hybrid.create_dataset("ivar", data=simulator.ivar_hybrid[idcs])
        grp_hybrid.create_dataset("mean-trans-flux", data=simulator.mean_trans_hybrid[idcs])

    grp_meta.attrs["fwhm"] = simulator.fwhm
    grp_meta.attrs["dv-fine"] = simulator.dvpix
    grp_meta.attrs["dv-coarse"] = simulator.dvpix_red
    grp_meta.attrs["npca"] = simulator.npca
    grp_meta.attrs["nskew"] = simulator.nskew
    grp_meta.attrs["wave-split"] = simulator.wave_split

    # add redshifts and magnitudes to the training/validation/test groups
    grp_traindata.create_dataset("redshifts", data=simulator.redshifts[train_idcs])
    grp_traindata.create_dataset("logLv", data=simulator.logLv_samp[train_idcs])
    #grp_traindata.create_dataset("mags", data=simulator.mags[train_idcs])
    grp_validdata.create_dataset("redshifts", data=simulator.redshifts[valid_idcs])
    grp_validdata.create_dataset("logLv", data=simulator.logLv_samp[valid_idcs])
    #grp_validdata.create_dataset("mags", data=simulator.mags[valid_idcs])
    grp_testdata.create_dataset("redshifts", data=simulator.redshifts[test_idcs])
    grp_testdata.create_dataset("logLv", data=simulator.logLv_samp[test_idcs])
    #grp_testdata.create_dataset("mags", data=simulator.mags[test_idcs])

    print ("Created file at {}".format(filename))

    return f


def constructTransmissionTemplates(simulator, filename):
    '''
    Construct an HDF5 file containing the mean transmission curves on a (z, logLv) grid.

    @param simulator: CombinedSimulations instance
    @param filename: str
    @return:
    '''

    f = h5py.File(filename, "w")

    # create groups for fine and hybrid grid
    grp_fine = f.create_group("fine-grid")
    grp_hybrid = f.create_group("hybrid-grid")

    grp_fine.create_dataset("mean-trans", data=simulator.trans_templates)
    grp_fine.create_dataset("wave-fine", data=simulator.wave_rest)

    grp_hybrid.create_dataset("mean-trans", data=simulator.trans_templates_hybrid)
    grp_hybrid.create_dataset("wave-hybrid", data=simulator.wave_hyrbid)

    # also store the redshift midpoints and logLv midpoints
    f.create_dataset("z-mids", simulator.z_mids)
    f.create_dataset("logLv-mids", simulator.logLv_mids)

    print ("Created file at {}".format(filename))

    return f