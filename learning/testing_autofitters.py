import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
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

        '''
        # convert flux and true continuum to numpy arrays for plotting purposes
        flux = spectra.flux.detach().numpy()
        cont = spectra.cont.detach().numpy()
        '''

        # get the normalised true continuum
        norm_factor = self.flux_norm / self.spectra.flux
        self.cont_norm = (norm_factor * self.spectra.cont).detach().numpy()

        self.wave_grid = spectra.wave_grid


class AutofitterPredictedSpectra(AutofitterPredictions):
    def __init__(self, spectra, model="autofitter"):
        super(AutofitterPredictedSpectra, self).__init__(spectra, model)

    def random_index(self, size=1):

        rand_indx = np.random.randint(0, len(self.spectra), size)

        return rand_indx


    def create_figure(self, figsize=(7,5), dpi=320):

        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.axes = []

        self.fig.suptitle("Model: "+self.model)

        return self.fig


    def plot(self, index, wave_lims=None, figsize=(7,5), dpi=320,
             subplotloc=111, alpha=0.7, contpredcolor="darkred"):

        try:
            fig = self.fig
        except:
            fig = self.create_figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(subplotloc)

        if wave_lims is None:
            grid = self.wave_grid
            flux = self.flux_norm[index].squeeze()
            cont = self.cont_norm[index].squeeze()
            cont_pred = self.cont_norm_pred[index].squeeze()

        elif len(wave_lims) == 2:
            sel = (self.wave_grid > wave_lims[0]) & (self.wave_grid < wave_lims[1])
            grid = self.wave_grid[sel]
            print (sel.shape)
            print (self.flux_norm[index].shape)
            flux = self.flux_norm[index].squeeze()[sel]
            cont = self.cont_norm[index].squeeze()[sel]
            cont_pred = self.cont_norm_pred[index].squeeze()[sel]

        else:
            raise ValueError("Parameter 'wave_lims' must be an array-like object (wave_min, wave_max).")

        ax.plot(grid, flux, alpha=alpha, lw=.5, label="Mock spectrum")
        ax.plot(grid, cont, alpha=alpha, lw=2., label="True continuum")
        ax.plot(grid, cont_pred, alpha=alpha, lw=1, ls="--",
                c=contpredcolor, label="Predicted continuum")

        ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"Normalised flux")
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        ax.grid(which="minor", linewidth=.1, alpha=.3, color="grey")
        ax.set_title(r"Results for test spectrum " + str(index + 1))

        self.axes.append(ax)

        return ax


    def show_figure(self):

        self.fig.tight_layout()
        self.fig.show()


class AutofitterRelResids(AutofitterPredictions):
    def __init__(self, spectra, model="autofitter"):
        '''

        @param spectra:
        @param model:
        '''
        super(AutofitterRelResids, self).__init__(spectra, model)

        self.rel_resid = (self.cont_norm - self.cont_norm_pred) / self.cont_norm
        self.mean_spec = np.mean(self.rel_resid, axis=0)
        self.std_spec = np.std(self.rel_resid, axis=0)
        self.mad_std_spec = mad_std(self.rel_resid, axis=0)

        self.mean_resid = np.mean(self.rel_resid)
        self.std_resid = np.std(self.rel_resid)
        self.mad_resid = mad_std(self.rel_resid)

        percent_min_1sig = np.percentile(self.rel_resid, 100.*norm.cdf(-1.), axis=0)
        percent_plu_1sig = np.percentile(self.rel_resid, 100.*norm.cdf(1.), axis=0)
        self.percentile_std = (percent_plu_1sig - percent_min_1sig) / 2.
        self.percentile_median = np.percentile(self.rel_resid, 50., axis=0)

        self.sigma_min = percent_min_1sig
        self.sigma_plu = percent_plu_1sig


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
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        ax.grid(which="minor", linewidth=.1, alpha=.3, color="grey")
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"$(F_\textrm{true} - F_\textrm{pred}) / F_\textrm{true}$")
        ax.set_title("Residuals relative to true continuum")

        return fig, ax


    def plot_percentiles(self, wave_lims=None):

        if wave_lims is None:
            grid = self.spectra.wave_grid
            median_spec = self.percentile_median
            perc_std_spec = self.percentile_std
            sigma_min = self.sigma_min
            sigma_plu = self.sigma_plu

        elif len(wave_lims) == 2:
            sel = (self.spectra.wave_grid > wave_lims[0]) & (self.spectra.wave_grid < wave_lims[1])
            grid = self.spectra.wave_grid[sel]
            median_spec = self.percentile_median[sel]
            perc_std_spec = self.percentile_std[sel]
            sigma_min = self.sigma_min[sel]
            sigma_plu = self.sigma_plu[sel]

        else:
            raise ValueError("Parameter 'wave_lims' must be an array-like object (wave_min, wave_max).")

        fig, ax = plt.subplots(figsize=(7,5), dpi=320)
        ax.plot(grid, median_spec, label="Median", color="black")
        ax.fill_between(grid, sigma_min, sigma_plu, alpha=0.3,\
                        label=r"68\% interval", color="tab:orange")

        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"$(F_\textrm{true} - F_\textrm{pred}) / F_\textrm{true}$")
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


class AutofitterCorrelationMatrix(AutofitterRelResids):
    def __init__(self, spectra, model="autofitter"):
        super(AutofitterCorrelationMatrix, self).__init__(spectra, model)

        diff = self.rel_resid - self.mean_spec
        covar_delta = 1 / (self.cont_norm.shape[0] - 1) * np.matmul(diff.T, diff)
        corr_delta = covar_delta / np.sqrt(np.outer(np.diag(covar_delta), np.diag(covar_delta)))
        self.matrix = corr_delta


    def show(self, wave_lims=None):

        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=320)
        self.im = self.ax.pcolormesh(self.wave_grid, self.wave_grid, self.matrix, cmap="bwr",
                                     shading="nearest", vmin=-1.0, vmax=1.0)
        self.ax.set_aspect("equal")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="Residual correlation")
        self.ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_ylabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_title("Correlation matrix of residuals")

        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        if wave_lims is None:
            pass

        elif len(wave_lims) == 2:
            self.ax.set_xlim(*wave_lims)
            self.ax.set_ylim(*wave_lims)

        else:
            raise ValueError("Parameter 'wave_lims' must be an array-like object (wave_min, wave_max).")

        self.fig.show()

        return self.fig, self.ax


    def savefig(self, fullfilename):
        '''Save the figure to a file with full path fullfilename. Only works if the figure is already created.'''

        try:
            self.fig.savefig(fullfilename)
        except:
            print ("Could not save a figure. Does it exist yet?")