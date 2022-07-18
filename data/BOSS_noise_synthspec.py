'''File for generating a training sample of QSO spectra with empirical noise vectors from BOSS.
The spectra are saved in an HDF5 file, on three grids: the hybrid grid, a uniform fine grid and a uniform coarse grid.
The fine pixels have log(wav) spacing 1e-4; the coarse pixels have velocity width 500 km/s.
No running-median smoothed spectrum is saved.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import h5py
from dw_inference.sdss.utils import get_wave_grid
from dw_inference.simulator.lognormal.lognormal_new import F_onorbe
from dw_inference.simulator.utils import get_blu_red_wave_grid
from linetools.lists.linelist import LineList
from dw_inference.simulator.proximity.proximity import Proximity
from load_data import normalise_spectra
from scipy import interpolate
from pypeit.utils import fast_running_median
from qso_fitting.data.utils import rebin_spectra
import astropy.constants as const
from matplotlib.ticker import AutoMinorLocator
from data.empirical_noise import rebinNoiseVectors, interpBadPixels

import os
print ("SPECDB environment: {}".format(os.getenv("SPECDB")))

plt.rcParams["font.family"] = "serif"

# simulate the continua
strong_lines = LineList("Strong", verbose=False)
lya_1216 = strong_lines["HI 1215"]
lyb_1025 = strong_lines["HI 1025"]
wave_1216 = lya_1216["wrest"].value
wave_1025 = lyb_1025["wrest"].value

'''
wave_min = 980.0
wave_max = 2040.0
'''
wave_min = 1000.
wave_max = 1970.
fwhm = 131.4   # approximate BOSS FWHM (Smee+ 2013)
dloglam = 1.0e-4
c_light = (const.c.to("km/s")).value
dvpix = dloglam * c_light * np.log(10)
wave_rest = get_wave_grid(wave_min, wave_max, dvpix)
mags = np.full(5, 18.5)
z_qso = 3.5

# load empirical noise vectors
zmin = z_qso - 0.01
zmax = z_qso + 0.01
flux_boss, ivar_boss, gpm_boss = rebinNoiseVectors(zmin, zmax, wave_rest)

print ("Number of negative/zero-valued ivar values:", np.sum(ivar_boss <= 0))

fig0, ax0 = plt.subplots(dpi=240)
ax0.hist(ivar_boss[ivar_boss > 0])
fig0.suptitle("Positive BOSS ivar values")
ax0.set_xlabel("ivar noise")
ax0.set_ylabel("Occurrences")
fig0.show()

iforest = (wave_rest > wave_1025) & (wave_rest < wave_1216)
z_lya = wave_rest[iforest]*(1.0 + z_qso)/wave_1216 - 1.0
mean_flux_z = F_onorbe(z_lya)
true_mean_flux = np.mean(mean_flux_z)
mean_flux_range_old = np.clip([true_mean_flux-0.1, true_mean_flux+0.1], 0.01, 1.0)
mean_flux_range = np.clip([true_mean_flux-0.0001, true_mean_flux+0.0001], 0.01, 1.0)
print ("Old mean flux range:", mean_flux_range_old)
print ("New mean flux range:", mean_flux_range)

nskew = 1000
npca = 10

pcafilename = 'COARSE_PCA_150_1000_2000_forest.pkl' # File holding (the old) PCA vectors
#nF = 10 # Number of mean flux
#nlogL = 5
L_rescale = 1.
nF = 2
nlogL = 2
L_rescale_range = (L_rescale-1e-4, L_rescale+1e-4)

mean_flux_vec = mean_flux_range[0] + (mean_flux_range[1] - mean_flux_range[0]) * np.arange(nF) / (nF - 1)
L_rescale_vec = L_rescale_range[0] + (L_rescale_range[1] - L_rescale_range[1]) * np.arange(nlogL) / (nlogL - 1)

print ("mean_flux_vec:", mean_flux_vec)
print ("L_rescale_vec:", L_rescale_vec)

pcafile = '/net/vdesk/data2/buiten/MRP2/Data/' + pcafilename
Prox = Proximity(wave_rest, fwhm, z_qso, mags, nskew, mean_flux_range, nF, npca, pcafile, nlogL=nlogL,
                 L_rescale_range=L_rescale_range)

# set the number of spectra to generate
nsamp = 25000

theta = Prox.sample_theta(nsamp)

# first column sets mean flux
# second column sets L_rescale
# we can manually set these to the single value we want
theta[:,0] = true_mean_flux
theta[:,1] = L_rescale

print ("Shape of theta:", theta.shape)
print ("theta:", theta)

# simulate the continua and fluxes
cont_prox, flux_prox = Prox.simulator(theta, replace=(nsamp > nskew), ivar=None)

# extract the transmission and compute the mean before adding noise
trans = flux_prox / cont_prox
mean_trans = np.mean(trans, axis=0)

# normalise to one at 1280 \AA
flux_norm, cont_norm = normalise_spectra(wave_rest, flux_prox, cont_prox)

# assign empirical noise vectors to generated spectra at random
rand_idx = np.random.randint(0, len(ivar_boss), size=len(flux_norm))

ivar_rand = np.zeros_like(cont_norm)
for i in range(len(cont_norm)):
    ivar_rand[i] = ivar_boss[rand_idx[i]]

sigma_vectors = np.sqrt(1 / ivar_rand)

# draw actual noise terms from the noise vectors
noise_terms = np.zeros_like(cont_norm)
for i in range(nsamp):
    noise_terms[i] = np.random.normal(0, sigma_vectors[i], size=sigma_vectors.shape[-1])

flux_norm_noisy = flux_norm + noise_terms

mean_trans_noise = np.mean(flux_norm_noisy / cont_norm, axis=0)

# create a hybrid grid
dvpix_red = 500.0
gpm_norm = None

# set the wavelength of the blue-red split
wave_split = wave_1216

wave_grid, dvpix_diff, ipix_blu, ipix_red = get_blu_red_wave_grid(wave_min, wave_max, wave_split, dvpix, dvpix_red)

# also create a fully coarse grid
wave_coarse = get_wave_grid(wave_min, wave_max, dvpix_red)

# interpolate the continuum and rebin the absorption spectra & noise vectors
cont_blu_red = interpolate.interp1d(wave_rest, cont_norm, kind="cubic", bounds_error=False, fill_value="extrapolate",
                                    axis=1)(wave_grid)
cont_coarse = interpolate.interp1d(wave_rest, cont_norm, kind="cubic", bounds_error=False, fill_value="extrapolate",
                                   axis=1)(wave_coarse)

flux_blu_red, ivar_rebin, gpm_rebin, count_rebin = rebin_spectra(wave_grid, wave_rest, flux_norm_noisy,
                                                                 ivar_rand, gpm=gpm_norm)
flux_coarse, ivar_coarse, gpm_coarse, count_coarse = rebin_spectra(wave_coarse, wave_rest, flux_norm_noisy,
                                                                   ivar_rand, gpm=gpm_norm)

# properly rebin the hybrid-grid noise vectors (i.e. with interpolation)
ivar_rebin = interpBadPixels(wave_grid, ivar_rebin, gpm_rebin)
ivar_coarse = interpBadPixels(wave_coarse, ivar_coarse, gpm_coarse)

sigma_rebin = 1 / np.sqrt(ivar_rebin)
sigma_coarse = 1 / np.sqrt(ivar_coarse)

print ("Number of negative ivar pixels:", np.sum(ivar_rebin <= 0))
print ("Number of negative ivar pixels on coarse grid:", np.sum(ivar_rebin <= 0))

#mean_trans_rebin = np.mean(flux_blu_red / cont_blu_red, axis=0)
#mean_trans_coarse = np.mean(flux_coarse / cont_coarse, axis=0)

# interpolate the (noiseless) mean transmission
mean_trans_coarse = interpolate.interp1d(wave_rest, mean_trans, kind="cubic", bounds_error=False,
                                         fill_value="extrapolate")(wave_coarse)
mean_trans_hybrid = interpolate.interp1d(wave_rest, mean_trans, kind="cubic", bounds_error=False,
                                         fill_value="extrapolate")(wave_grid)

# plot a random example and its noise vector

idx = np.random.randint(0, len(cont_norm))

fig, ax = plt.subplots(dpi=240)
ax.plot(wave_coarse, cont_coarse[idx], alpha=0.7, label="Continuum", c="tab:orange")
ax.plot(wave_coarse, flux_coarse[idx], alpha=0.5, label="Noisy spectrum", lw=.5, c="tab:blue")
ax.plot(wave_coarse, sigma_coarse[idx], alpha=.7, color="tab:green", lw=.5, label="Noise vector")
ax.set_xlabel("Rest-frame wavelength ($\AA$)")
ax.set_ylabel("Normalised flux")
ax.legend()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(which="major")
ax.grid(which="minor", linewidth=.1, alpha=.3, color="grey")
fig.suptitle("Synthetic Spectrum {} with BOSS Noise".format(idx))
ax.set_title("Coarse grid")
fig.show()

#figpath = "/net/vdesk/data2/buiten/MRP2/misc-figures/hetsced-noise/BOSS-noise/"
#fig.savefig("{}synthspec-BOSSnoise-example{}.png".format(figpath, idx))

# plot the mean of the ivar across the spectrum
ivar_mean_spec = np.mean(ivar_coarse, axis=0)
fig2, ax2 = plt.subplots(dpi=240)
ax2.plot(wave_coarse, ivar_mean_spec)
ax2.set_xlabel(r"Rest-frame wavelength ($\AA$)")
ax2.set_ylabel(r"Normalised ivar (a.u.)")
ax2.set_title("Mean of ivar noise across spectrum")
fig2.suptitle("Mean of inverse variance")
fig2.show()

# plot the mean transmitted flux vs wavelength
# make the plot on three different grids
fig3, ax3 = plt.subplots(dpi=240)
ax3.plot(wave_rest, mean_trans, label="Fine grid", alpha=.7, lw=.5)
ax3.plot(wave_coarse, mean_trans_coarse, label="Coarse grid", alpha=.7)
ax3.plot(wave_grid, mean_trans_hybrid, label="Hybrid grid", alpha=.5, lw=.5)
#ax3.plot(wave_rest, mean_trans_noise, label="With noise", alpha=.7)
#ax3.plot(wave_grid, mean_trans_rebin, label="Hybrid grid", alpha=.5, lw=.5)
ax3.set_xlabel(r"Rest-frame wavelength ($\AA$)")
ax3.set_ylabel(r"$\langle F_{abs} / F_{cont} \rangle$")
fig3.suptitle("Mean Trasmitted Flux")
ax3.set_title("Without noise")
ax3.legend()
fig3.show()

# split the data into a training/validation/test sets
train_frac = 0.9
valid_frac = 0.5 * (1 - train_frac)
test_frac = 1 - train_frac - valid_frac

rng = np.random.default_rng()
all_idcs = np.arange(0, nsamp)
train_idcs = rng.choice(all_idcs, size=int(train_frac * nsamp), replace=False)
valid_idcs = rng.choice(np.delete(all_idcs, train_idcs), size=int(valid_frac * nsamp), replace=False)
test_idcs = np.delete(all_idcs, np.concatenate((train_idcs, valid_idcs)))

# store everything in an hdf5 file
filepath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
filename = "{}synthspec_BOSSlike_npca{}_z{}_large.hdf5".format(filepath, npca, z_qso)
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

grp_meta.create_dataset("wave-fine", data=wave_rest)
grp_meta.create_dataset("wave-coarse", data=wave_coarse)
grp_meta.create_dataset("wave-hybrid", data=wave_grid)

# add the indexed spectra to the right group
grps = [[grp_fine_train, grp_coarse_train, grp_hybrid_train], [grp_fine_valid, grp_coarse_valid, grp_hybrid_valid],
        [grp_fine_test, grp_coarse_test, grp_hybrid_test]]

for (idcs, [grp_fine, grp_coarse, grp_hybrid]) in zip([train_idcs, valid_idcs, test_idcs], grps):

    nsamp_set = len(idcs)

    grp_fine.create_dataset("cont", data=cont_norm[idcs])
    grp_fine.create_dataset("flux", data=flux_norm_noisy[idcs])
    grp_fine.create_dataset("ivar", data=ivar_rand[idcs])
    grp_fine.create_dataset("mean-trans-flux", data=np.full((nsamp_set, cont_norm.shape[-1]), mean_trans))
    # also save the noiseless absorption spectrum
    grp_fine.create_dataset("noiseless-flux", data=flux_norm[idcs])

    grp_coarse.create_dataset("cont", data=cont_coarse[idcs])
    grp_coarse.create_dataset("flux", data=flux_coarse[idcs])
    grp_coarse.create_dataset("ivar", data=ivar_coarse[idcs])
    grp_coarse.create_dataset("mean-trans-flux", data=np.full((nsamp_set, cont_coarse.shape[-1]), mean_trans_coarse))

    grp_hybrid.create_dataset("cont", data=cont_blu_red[idcs])
    grp_hybrid.create_dataset("flux", data=flux_blu_red[idcs])
    grp_hybrid.create_dataset("ivar", data=ivar_rebin[idcs])
    grp_hybrid.create_dataset("mean-trans-flux", data=np.full((nsamp_set, cont_blu_red.shape[-1]), mean_trans_hybrid))

grp_meta.attrs["fwhm"] = fwhm
grp_meta.attrs["dv-fine"] = dvpix
grp_meta.attrs["dv-coarse"] = dvpix_red
grp_meta.attrs["npca"] = npca
grp_meta.attrs["nskew"] = nskew

# add redshifts and magnitudes to the training/validation/test groups
grp_traindata.create_dataset("redshifts", data=np.full(len(train_idcs), z_qso))
grp_traindata.create_dataset("mags", data=mags[train_idcs])
grp_validdata.create_dataset("redshifts", data=np.full(len(valid_idcs), z_qso))
grp_validdata.create_dataset("mags", data=mags[valid_idcs])
grp_testdata.create_dataset("redshifts", data=np.full(len(test_idcs), z_qso))
grp_testdata.create_dataset("mags", data=mags[test_idcs])

#grp_meta.create_dataset("redshifts", data=np.full(nsamp, z_qso))
#grp_meta.create_dataset("mags", data=mags)

f.close()

print ("Saved file to {}".format(filename))