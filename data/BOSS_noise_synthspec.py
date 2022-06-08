'''File for generating a training sample of QSO spectra with the
Ly-alpha forest and with homoscedastic noise added in, saved on a uniform grid.'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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
#sampling = 2.0
#dvpix = fwhm/sampling
dloglam = 1.0e-4
c_light = (const.c.to("km/s")).value
dvpix = dloglam * c_light * np.log(10)
wave_rest = get_wave_grid(wave_min, wave_max, dvpix)
mags = np.full(5, 18.5)
z_qso = 2.8

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
cont_prox, flux_prox = Prox.simulator(theta, replace=(nsamp > nskew),\
                                      ivar=None)

zmin = z_qso - 0.01
zmax = z_qso + 0.01
# normalise to one at 1280 \AA
flux_norm, cont_norm = normalise_spectra(wave_rest, flux_prox, cont_prox)

# now add noise from empirical noise vectors
flux_boss, ivar_boss, gpm_boss = rebinNoiseVectors(zmin, zmax, wave_rest)

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

# smooth the flux before regridding
flux_smooth = np.zeros(flux_norm_noisy.shape)
for i, F in enumerate(flux_norm_noisy):
    flux_smooth[i,:] = fast_running_median(F, window_size=20)

gpm_norm = None

# smooth the flux before regridding
flux_smooth = np.zeros(flux_norm_noisy.shape)
for i, F in enumerate(flux_norm_noisy):
    flux_smooth[i,:] = fast_running_median(F, window_size=20)

# propagate the noise vector on the smoothed spectrum
sigma_smooth = sigma_vectors/np.sqrt(20)
ivar_smooth = 1/sigma_smooth**2

# interpolate onto the hybrid grid
dvpix_red = 500.0
gpm_norm = None

# set the wavelength of the blue-red split
wave_split = wave_1216

wave_grid, dvpix_diff, ipix_blu, ipix_red = get_blu_red_wave_grid(wave_min, wave_max,\
                                                                  wave_split, dvpix, dvpix_red)

cont_blu_red = interpolate.interp1d(wave_rest, cont_norm, kind="cubic", bounds_error=False,\
                                    fill_value="extrapolate", axis=1)(wave_grid)

flux_blu_red, ivar_rebin, gpm_rebin, count_rebin = rebin_spectra(wave_grid,\
                                                                 wave_rest,\
                                                                 flux_norm_noisy,\
                                                                 ivar_rand,\
                                                                 gpm=gpm_norm)
flux_smooth_blu_red, _, _, _ = rebin_spectra(wave_grid, wave_rest, flux_smooth,\
                                             ivar_smooth, gpm=gpm_norm)

# properly rebin the hybrid-grid noise vectors (i.e. with interpolation)
ivar_rebin = interpBadPixels(wave_grid, ivar_rebin, gpm_rebin)
sigma_rebin = 1 / np.sqrt(ivar_rebin)

print ("Number of negative ivar pixels:", np.sum(ivar_rebin <= 0))

# plot a random example and its noise vector

idx = np.random.randint(0, len(cont_norm))

fig, ax = plt.subplots(dpi=240)
ax.plot(wave_grid, cont_blu_red[idx], alpha=0.7, label="Continuum")
ax.plot(wave_grid, flux_blu_red[idx], alpha=0.5, label="Noisy spectrum", lw=.5)
#ax.plot(wave_grid, flux_smooth_blu_red[idx], alpha=0.7, color="navy", ls="--",\
#        label="Smoothed flux", lw=.5)
ax.plot(wave_grid, sigma_rebin[idx], alpha=.7, color="darkred", lw=.5, label="Noise vector")
ax.set_xlabel("Rest-frame wavelength ($\AA$)")
ax.set_ylabel("Normalised flux")
ax.legend()
ax.xaxis.set_minor_locator(AutoMinorLocator(5))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))
ax.grid(which="major")
ax.grid(which="minor", linewidth=.1, alpha=.3, color="grey")
fig.suptitle("Noiseless continuum vs noisy spectrum with Ly-$\\alpha$ forest")
ax.set_title("Empirical BOSS noise (spectrum {})".format(idx))
fig.show()

figpath = "/net/vdesk/data2/buiten/MRP2/misc-figures/hetsced-noise/BOSS-noise/"
fig.savefig("{}synthspec-BOSSnoise-example{}.png".format(figpath, idx))

# plot the mean of the ivar across the spectrum
ivar_mean_spec = np.mean(ivar_rebin, axis=0)
fig2, ax2 = plt.subplots(dpi=240)
ax2.plot(wave_grid, ivar_mean_spec)
ax2.set_xlabel(r"Rest-frame wavelength ($\AA$)")
ax2.set_ylabel(r"Normalised ivar (a.u.)")
ax2.set_title("Mean of ivar noise across spectrum")
fig2.show()

# save the grid, continuum and noisy continuum to an array
savearray = np.zeros((nsamp, len(wave_rest), 5))
savearray[:,:,0] = wave_rest

savearray_regridded = np.zeros((nsamp, len(wave_grid), 5))
savearray_regridded[:,:,0] = wave_grid

for i in range(nsamp):
    savearray[i,:,1] = cont_norm[i,:]
    savearray[i,:,2] = flux_norm_noisy[i,:]
    savearray[i,:,3] = flux_smooth[i,:]
    savearray[i,:,4] = ivar_rand[i,:]

    savearray_regridded[i,:,1] = cont_blu_red[i,:]
    savearray_regridded[i,:,2] = flux_blu_red[i,:]
    savearray_regridded[i,:,3] = flux_smooth_blu_red[i,:]
    savearray_regridded[i,:,4] = ivar_rebin[i,:]

savepath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
np.save(savepath+"forest_spectra_BOSSnoise_npca"+str(npca)+"BOSS-grid.npy",\
        savearray)
print ("Array saved.")

np.save(savepath+"forest_spectra_BOSSnoise_npca"+str(npca)+"BOSS-regridded.npy",
        savearray_regridded)
print ("Regridded array saved.")