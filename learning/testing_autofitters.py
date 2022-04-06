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

        rand_indx = np.random.rand(0, len(self.spectra), size)

        return rand_indx


    def create_figure(self, figsize=(7,5), dpi=320):

        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.axes = []

        self.fig.suptitle("Model: "+self.model)


    def plot(self, index, wave_lims=None, figsize=(7,5), dpi=320,
             subplotloc=111, alpha=0.7, contpredcolor="darkred"):

        try:
            fig = self.fig
        except:
            fig = self.create_figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(subplotloc)

        if wave_lims is None:
            grid = self.wave_grid
            flux = self.flux_norm[index]
            cont = self.cont_norm[index]
            cont_pred = self.cont_norm_pred[index]

        elif len(wave_lims) == 2:
            sel = (self.wave_grid > wave_lims[0]) & (self.wave_grid < wave_lims[1])
            grid = self.wave_grid[sel]
            flux = self.flux_norm[sel][index]
            cont = self.cont_norm[sel][index]
            cont_pred = self.cont_norm_pred[sel][index]

        else:
            raise ValueError("Parameter 'wave_lims' must be an array-like object (wave_min, wave_max).")

        ax.plot(grid, flux, alpha=alpha, lw=1., label="Mock spectrum")
        ax.plot(grid, cont, alpha=alpha, lw=2., label="True continuum")
        ax.plot(grid, cont_pred, alpha=alpha, lw=1, ls="--",
                c=contpredcolor, label="Predicted continuum")

        ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"Normalised flux")
        ax.legend()
        ax.grid()
        ax.set_title(r"Results for test spectrum " + str(index + 1))

        self.axes.append(ax)

        return ax


    def show_figure(self):

        self.fig.tight_layout()
        self.fig.show()


class AutofitterRelResids(AutofitterPredictions):
    def __init__(self, spectra, model="autofitter"):
        super(AutofitterRelResids, self).__init__(spectra, model)

        self.rel_resid = (self.cont_norm - self.cont_norm_pred) / self.cont_norm
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