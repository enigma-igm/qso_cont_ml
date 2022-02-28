'''File for generating a training sample of QSO spectra with the
Ly-alpha forest and with homoscedastic noise added in.'''

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
from qso_fitting.data.sdss.utils import rebin_spectra

plt.rcParams["font.family"] = "serif"

# simulate the continua
strong_lines = LineList("Strong", verbose=False)
lya_1216 = strong_lines["HI 1215"]
lyb_1025 = strong_lines["HI 1025"]
wave_1216 = lya_1216["wrest"].value
wave_1025 = lyb_1025["wrest"].value

wave_min = 1020.0
wave_max = 1970.0
fwhm = 150.0
sampling = 2.0
dvpix = fwhm/sampling
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
#cont_prox = Prox.simulator_continuum(theta)
#cont_norm, _ = normalise_spectra(wave_rest, cont_prox, cont_prox)

# simulate the Ly-alpha forest and multiply by the continuum
#t_prox = Prox.simulator_lya(theta)
#flux = t_prox*cont_prox

# normalise to one at 1280 \AA
flux_norm, cont_norm = normalise_spectra(wave_rest, flux_prox, cont_prox)

# now generate homoscedastic noise and add it to the flux
SN = 10
std_noise1280 = 1/SN
gauss = norm(scale=std_noise1280)
noise_vector = gauss.rvs(size=cont_norm.shape)
flux_norm_noisy = flux_norm + noise_vector

# smooth the flux before regridding
flux_smooth = np.zeros(flux_norm_noisy.shape)
for i, F in enumerate(flux_norm_noisy):
    flux_smooth[i,:] = fast_running_median(F, window_size=20)

# propagate the noise vector on the smoothed spectrum
sigma_spec = std_noise1280*np.ones(cont_norm.shape)
sigma_smooth = sigma_spec/np.sqrt(20)

# interpolate onto the hybrid grid
dvpix_red = 500.0
gpm_norm = None

# set the wavelength of the blue-red split
#wave_split = wave_1216
wave_split = 1230.

wave_grid, dvpix_diff, ipix_blu, ipix_red = get_blu_red_wave_grid(wave_min, wave_max,\
                                                                  wave_split, dvpix, dvpix_red)
cont_blu_red = interpolate.interp1d(wave_rest, cont_norm, kind="cubic", bounds_error=False,\
                                    fill_value="extrapolate", axis=1)(wave_grid)
flux_blu_red, ivar_rebin, gpm_rebin, count_rebin = rebin_spectra(wave_grid,\
                                                                 wave_rest,\
                                                                 flux_norm_noisy,\
                                                                 1/sigma_spec**2,\
                                                                 gpm=gpm_norm)
flux_smooth_blu_red, _, _, _ = rebin_spectra(wave_grid, wave_rest, flux_smooth,\
                                             1/sigma_smooth**2, gpm=gpm_norm)

#flux_blu_red = interpolate.interp1d(wave_rest, flux_norm_noisy, kind="cubic", bounds_error=False,\
#                                    fill_value="extrapolate", axis=1)(wave_grid)
#flux_smooth_blu_red = interpolate.interp1d(wave_rest, flux_smooth, kind="cubic", bounds_error=False,\
#                                           fill_value="extrapolate", axis=1)(wave_grid)

# plot the first example
fig, ax = plt.subplots()
ax.plot(wave_grid, cont_blu_red[0], alpha=0.7, label="Regridded continuum")
ax.plot(wave_grid, flux_blu_red[0], alpha=0.7, label="Regridded noisy spectrum")
ax.plot(wave_grid, flux_smooth_blu_red[0], alpha=0.7, color="navy", ls="--",\
        label="Regridded smoothed flux")
ax.axvline(wave_split, ls=":", c="black", lw="1", label="Blue-red split")
ax.set_xlabel("Rest-frame wavelength ($\AA$)")
ax.set_ylabel("Normalised flux")
ax.legend()
fig.suptitle("Noiseless continuum vs noisy spectrum with Ly-$\\alpha$ forest")
ax.set_title("Homoscedastic noise with $\sigma = 0.1$")
fig.show()

# save the grid, continuum and noisy continuum to an array
savearray = np.zeros((nsamp, len(wave_grid), 4))
savearray[:,:,0] = wave_grid

for i in range(nsamp):
    savearray[i,:,1] = cont_blu_red[i,:]
    savearray[i,:,2] = flux_blu_red[i,:]
    savearray[i,:,3] = flux_smooth_blu_red[i,:]

savepath = "/net/vdesk/data2/buiten/MRP2/pca-sdss-old/"
np.save(savepath+"forest_spectra_with_noiseSN"+str(SN)+"_regridded_npca"+str(npca)+"smooth-window20_split"+str(int(wave_split))+".npy",\
        savearray)
print ("Array saved.")