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
z_qso = 2.8

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
mean_flux_range = np.clip([true_mean_flux-0.1, true_mean_flux+0.1], 0.01, 1.0)

nskew = 1000
npca = 10

pcafilename = 'COARSE_PCA_150_1000_2000_forest.pkl' # File holding (the old) PCA vectors
nF = 10 # Number of mean flux
nlogL = 5
pcafile = '/net/vdesk/data2/buiten/MRP2/Data/' + pcafilename
Prox = Proximity(wave_rest, fwhm, z_qso, mags, nskew, mean_flux_range, nF, npca, pcafile, nlogL=nlogL)

# set the number of spectra to generate
nsamp = 25000

theta = Prox.sample_theta(nsamp)

# simulate the continua and fluxes
cont_prox, flux_prox = Prox.simulator(theta, replace=(nsamp > nskew), ivar=None)

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
ax2.plot(wave_grid, ivar_mean_spec)
ax2.set_xlabel(r"Rest-frame wavelength ($\AA$)")
ax2.set_ylabel(r"Normalised ivar (a.u.)")
ax2.set_title("Mean of ivar noise across spectrum")
fig2.suptitle("Mean of inverse variance")
fig2.show()

# store everything in an hdf5 file
filepath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
filename = "{}synthspec_BOSSlike_npca{}_z{}_large.hdf5".format(filepath, npca, z_qso)
f = h5py.File(filename, "w")

struct_arr_fine = np.zeros((3, nsamp, len(wave_rest)))
struct_arr_coarse = np.zeros((3, nsamp, len(wave_coarse)))
struct_arr_hybrid = np.zeros((3, nsamp, len(wave_grid)))

#for i in range(3):
#    struct_arr_fine[i] =

dset_fine = f.create_dataset("uniform-fine",)
dset_coarse = f.create_dataset("uniform-coarse")
dset_hybrid = f.create_dataset("hybrid")

f.close()