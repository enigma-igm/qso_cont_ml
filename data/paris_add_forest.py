import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from dw_inference.sdss.utils import get_wave_grid
from dw_inference.simulator.lognormal.lognormal_new import F_onorbe
from dw_inference.simulator.utils import get_blu_red_wave_grid
from linetools.lists.linelist import LineList
from qso_fitting.data.sdss.paris import read_paris_continua
from dw_inference.simulator.proximity.proximity import Proximity

plt.rcParams["font.family"] = "serif"

wave_norm, cont_norm, flux_norm, ivar_norm, paris_out = read_paris_continua()
z_qso = np.mean(paris_out["z"])

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
#z_qso = 2.8
mags = np.full(5, 18.5)

iforest = (wave_rest > wave_1025) & (wave_rest < wave_1216)
z_lya = wave_rest[iforest]*(1.0 + z_qso)/wave_1216 - 1.0
mean_flux_z = F_onorbe(z_lya)
true_mean_flux = np.mean(mean_flux_z)
mean_flux_range = np.clip([true_mean_flux-0.1, true_mean_flux+0.1], 0.01, 1.0)

nskew = 1000
npca = 15

pcafilename = 'COARSE_PCA_150_1000_2000_forest.pkl' # File holding (the old) PCA vectors
nF = 10 # Number of mean flux
nlogL = 5
pcafile = '/net/vdesk/data2/buiten/MRP2/Data/' + pcafilename
Prox = Proximity(wave_norm, fwhm, z_qso, mags, nskew, mean_flux_range, nF, npca, pcafile, nlogL=nlogL)

#nsamp = 1
nsamp = len(cont_norm) # Number of mock quasars to generate

print (wave_rest.shape)
print (mean_flux_z.shape)
print (true_mean_flux)

theta = Prox.sample_theta(nsamp)
t_prox = Prox.simulator_lya(theta)

testcont = np.ones(len(wave_rest))
testflux = t_prox*cont_norm

# now we want to interpolate onto the hybrid grid
dvpix_red = 500.0
wave_grid, dvpix_diff, ipix_blu, ipix_red = get_blu_red_wave_grid(wave_min, wave_max,\
                                                                  wave_1216, dvpix, dvpix_red)
cont_blu_red = interpolate.interp1d(wave_norm, cont_norm, kind="cubic", bounds_error=False,\
                                    fill_value="extrapolate", axis=1)(wave_grid)
flux_blu_red = interpolate.interp1d(wave_norm, testflux, kind="cubic", bounds_error=False,\
                                    fill_value="extrapolate", axis=1)(wave_grid)

fig, ax = plt.subplots()
ax.plot(wave_norm, testflux[0], alpha=0.7, label="Initial grid")
ax.plot(wave_grid, flux_blu_red[0], alpha=0.7, label="Hybrid grid")
ax.set_xlabel("Rest-frame wavelength ($\AA$)")
ax.set_ylabel("Normalised flux")
ax.legend()
fig.show()

# save the grid, the continuum and the noiseless flux to an array
savearray = np.zeros((nsamp, len(wave_grid), 3))
savearray[:,:,0] = wave_grid

for i in range(nsamp):
    savearray[i,:,1] = cont_blu_red[i,:]
    savearray[i,:,2] = flux_blu_red[i,:]

savepath = "/net/vdesk/data2/buiten/MRP2/code/qso_cont_ml/data/"
np.save(savepath+"paris_noiselessflux_regridded.npy", savearray)
print ("Array saved.")