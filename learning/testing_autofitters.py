import numpy as np
import matplotlib.pyplot as plt
from qso_fitting.data.sdss.sdss import autofit_continua, qsmooth_continua
from astropy.stats import mad_std
from scipy.stats import norm

import data.load_autofitter_datasets


class AutofitterPredictions:

    def __init__(self, spectra, model="autofitter"):

        if isinstance(spectra, data.load_autofitter_datasets.AutofitterSpectra):
            self.spectra = spectra
        else:
            raise TypeError("Parameter 'spectra' must be an instance of AutofitterSpectra.")

        if model=="autofitter":
            self.model = model
            self.flux_norm, self.ivar_norm, self.cont_norm_pred, self.cont_pred = autofit_continua(spectra.redshifts,
                                                                                                   spectra.wave_obs,
                                                                                                   spectra.flux,
                                                                                                   spectra.ivar)

        elif model=="Qsmooth":
            self.model = model
            self.flux_norm, self.ivar_norm, self.cont_norm_pred, self.cont_pred = qsmooth_continua(spectra.redshifts,
                                                                                                   spectra.wave_obs,
                                                                                                   spectra.flux,
                                                                                                   spectra.ivar)

        else:
            raise ValueError("Parameter 'model' should be either 'autofitter' or 'Qsmooth'.")

        # convert flux and true continuum to numpy arrays for plotting purposes
        self.flux = spectra.flux.detach().numpy()
        self.cont = spectra.cont.detach().numpy()


class AutofitterRelResids(AutofitterPredictions):
    def __init__(self, spectra, model="autofitter"):
        super(AutofitterRelResids, self).__init__(spectra, model)

        self.rel_resid = (self.cont - self.cont_pred) / self.cont
        self.mean_spec = np.mean(self.rel_resid, axis=0)
        self.std_spec = np.std(self.rel_resid, axis=0)
        self.mad_std_spec = mad_std(self.rel_resid, axis=0)

        self.mean_resid = np.mean(self.rel_resid)
        self.std_resid = np.std(self.rel_resid)
        self.mad_resid = mad_std(self.rel_resid)


class AutofitterResidualPlots(AutofitterRelResids):
    def __init__(self, spectra, model="autofitter"):
        super(AutofitterResidualPlots, self).__init__(spectra, model)


    def plot_means(self, show_std=False, wave_lims=None):

        if wave_lims is None:
            grid = self.spectra.wave_grid
            mean_spec = self.mean_spec
            std_spec = self.std_spec
            mad_std_spec = self.mad_std_spec

        elif len(wave_lims) == 2:
            sel = (self.spectra.wave_grid > wave_lims[0]) & (self.spectra.wave_grid < wave_lims[1])
            grid = self.spectra.wave_grid[sel]
            mean_spec = self.mean_spec[sel]
            std_spec = self.std_spec[sel]
            mad_std_spec = self.mad_std_spec[sel]

        else:
            raise ValueError("Parameter 'wave_lims' must be an array-like object (wave_min, wave_max).")

        fig, ax = plt.subplots(figsize=(7,5), dpi=320)
        ax.plot(grid, mean_spec, label="Mean", color="black")

        if show_std:
            ax.fill_between(grid, mean_spec-std_spec, mean_spec+std_spec, alpha=0.3,\
                            label="Standard deviation", color="tab:blue")
        ax.fill_between(grid, mean_spec-mad_std_spec, mean_spec+mad_std_spec, alpha=0.3,\
                        label="MAD standard deviation", color="tab:orange")

        ax.legend()
        ax.grid()
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel("$\\frac{F_{true} - F_{pred}}{F_{true}}$")
        ax.set_title("Residuals relative to true continuum")

        return fig, ax


    def resid_hist(self):

        fig, ax = plt.subplots()
        hist, bins, patches = ax.hist(self.rel_resid.flatten(), bins=100, density=True, range=(-0.3, 0.3),\
                                       label="mean={:5.3f}, std = {:5.3f}, 1.48*mad={:5.3f}".format(self.mean_resid,\
                                                                                                    self.std_resid,\
                                                                                                    self.mad_resid))
        bin_cen = (bins[:-1] + bins[1:])/2
        ax.plot(bin_cen, norm.pdf(bin_cen, loc=self.mean_resid, scale=self.mad_resid), label="Gaussian (with MAD std)")
        ax.set_xlabel("$\\frac{F_{true} - F_{pred}}{F_{true}}$")
        ax.set_ylabel("Probability density")
        ax.set_title("Residuals relative to true continuum")
        ax.legend()
        return fig, ax