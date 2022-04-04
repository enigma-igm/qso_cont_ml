from qso_fitting.data.sdss.sdss import autofit_continua, qsmooth_continua
from data.load_datasets import SynthSpectra
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

spectra = SynthSpectra(regridded=False, noise=True, boss=True)
spectra.add_channel_shape()
trainset, validset, testset = spectra.split()

n_qso = len(testset)
n_pix = len(testset.wave_grid)
wave_rest = np.zeros((n_qso, n_pix))
wave_rest[:] = testset.wave_grid
z = 2.8 * np.ones((n_qso,n_pix))
wave_obs = wave_rest * (1+z)

# specify the noise levels
sigma = 0.1 * np.ones((n_qso, n_pix))
ivar = 1 / sigma**2

# run the autofitters
flux_norm, ivar_norm, cont_norm, cont = autofit_continua(z[:,0], wave_obs, testset.flux[:,0,:],
                                                         ivar)
flux_norm_qsmooth, ivar_norm_qsmooth, cont_norm_qsmooth, cont_qsmooth = qsmooth_continua(z[:,0],
                                                                                         wave_obs,
                                                                                         testset.flux[:,0,:],
                                                                                         ivar)

# plot an example
fig, ax = plt.subplots(figsize=(7,5), dpi=240)
ax.plot(wave_obs[0], flux_norm[0], alpha=.8, label="Flux", lw=.5)
ax.plot(wave_obs[0], cont_norm[0], alpha=.8, label="Autofit continuum", c="darkred")
ax.plot(wave_obs[0], cont_norm_qsmooth[0], alpha=.8, label="QSmooth continuum", c="navy")
ax.plot(wave_obs[0], testset.cont[0,0], alpha=.8, label="True continuum")
ax.legend()
ax.set_xlabel(r"Observed wavelength ($\AA$)")
ax.set_ylabel(r"Normalised flux")
fig.show()

path = "/net/vdesk/data2/buiten/MRP2/misc-figures/autofitter-tests/"
fig.savefig(path+"autofitter-qsmoothfitter-synthspec-example2.png")

