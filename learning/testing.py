import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from utils.errorfuncs import relative_residuals, corr_matrix_relresids
from astropy.stats import mad_std
from scipy.stats import norm
from scipy.interpolate import interp1d
import torch
#from pypeit.utils import fast_running_median
from utils.grids import rest_BOSS_grid
from data.load_datasets import SynthSpectra


class ModelResults:
    '''Class for predicting the continua for the test set and converting everything to numpy arrays.
    Uses only global QuasarScalers, or no scalers at all. If interpolate==True, the output is interpolated
    onto a uniform grid.'''

    def __init__(self, testset, net, scaler_flux=None,\
                 scaler_cont=None, smooth=False, interpolate=False):

        '''TO DO: get the actual non-regridded simulator output as the uniform-grid continua.'''

        self.wave_grid = testset.wave_grid
        self.flux = testset.flux
        self.cont = testset.cont
        self.scaler_flux = scaler_flux
        self.scaler_cont = scaler_cont
        self.net = net
        self.smooth = smooth
        self.interpolate = interpolate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if scaler_flux is None:
            self.use_QSOScaler = False
        else:
            self.use_QSOScaler = True

        flux_tensor = torch.FloatTensor(self.flux).to(self.device)

        # smooth input for the final skip connection before applying the QSOScaler
        if smooth:
            try:
                self.flux_smooth = testset.flux_smooth
            except:
                from pypeit.utils import fast_running_median
                flux_smooth = np.zeros(self.flux.shape)
                for i in range(len(flux_smooth)):
                    flux_smooth[i] = fast_running_median(self.flux[i], 20)
                self.flux_smooth = flux_smooth

            input_smooth = torch.FloatTensor(self.flux_smooth).to(self.device)

        else:
            input_smooth = None

        if self.use_QSOScaler:
            input = self.scaler_flux.forward(flux_tensor)
            self.flux_scaled = input.cpu().detach().numpy()

            if self.smooth:
                input_smooth = self.scaler_flux.forward(input_smooth)

            # the exception allows the class to work with e.g. a convolutional U-Net
            try:
                res = self.net(input, smooth=self.smooth, x_smooth=input_smooth)
            except:
                res = self.net(input)

            res_descaled = self.scaler_cont.backward(res)
            res_np = res_descaled.cpu().detach().numpy()

        else:
            try:
                res = self.net(flux_tensor, smooth=self.smooth, x_smooth=input_smooth)
            except:
                res = self.net(flux_tensor)

            res_np = res.cpu().detach().numpy()
            self.flux_scaled = self.flux

        self.cont_pred_np = res_np
        self.cont_pred_scaled_np = res.cpu().detach().numpy()

        # also scale the true continuum
        if self.use_QSOScaler & (self.cont is not None):
            cont_true_scaled = self.scaler_cont.forward(torch.FloatTensor(self.cont).to(self.device))
            self.cont_true_scaled_np = cont_true_scaled.cpu().detach().numpy()
        else:
            self.cont_true_scaled_np = self.cont

        # extract the noise vectors if available
        if testset.ivar is not None:
            self.ivar = testset.ivar.cpu().detach().numpy()
            self.noise = 1 / np.sqrt(self.ivar)
            self.flux = testset.flux[:,0,:]
        else:
            self.ivar = None
            self.noise = None
            self.flux = testset.flux

        # interpolate the output onto a uniform grid, if desired
        if interpolate:

            # quick fix: simply assume that the spectra are the BOSS ones
            # and load the uniform-grid true continua
            uni_testset = SynthSpectra(regridded=False, test=True)

            if self.ivar is None:
                uni_testset.add_channel_shape()
            else:
                uni_testset.add_noise_channel()

            print ("Uniform-grid spectra loaded.")

            self.uni_cont = uni_testset.cont.cpu().detach().numpy()
            self.uni_wave_grid = uni_testset.wave_grid
            #self.uni_wave_grid = rest_BOSS_grid()
            print ("True continua and native grid extracted.")

            self.uni_cont_pred = interp1d(self.wave_grid, self.cont_pred_np, kind="cubic", axis=-1,
                                              bounds_error=False, fill_value="extrapolate")(self.uni_wave_grid)
            #self.uni_cont = interp1d(self.wave_grid, self.cont, kind="cubic", axis=-1,
            #                             bounds_error=False, fill_value="extrapolate")(self.uni_wave_grid)
            #self.uni_cont_pred_scaled = interp1d(self.wave_grid, self.cont_pred_scaled_np, kind="cubic", axis=-1,
            #                                     bounds_error=False, fill_value="extrapolate")(self.uni_wave_grid)
            #self.uni_cont_scaled = interp1d(self.wave_grid, self.cont_true_scaled_np, kind="cubic", axis=-1,
            #                                bounds_error=False, fill_value="extrapolate")(self.uni_wave_grid)
            self.uni_cont_pred_scaled = None
            self.uni_cont_scaled = None


            print ("Interpolated the predictions.")

        else:
            self.uni_wave_grid = None
            self.uni_cont_pred = None
            self.uni_cont = None
            self.uni_cont_pred_scaled = None
            self.uni_cont_scaled = None


class ModelResultsSpectra(ModelResults):
    '''Class for example spectra from the test set and the corresponding model predictions.
    TO DO: allow for interpolated plots (parameter interpolate currently does nothing).'''

    def __init__(self, testset, net, scaler_flux, scaler_cont, smooth=False, interpolate=False):
        super(ModelResultsSpectra, self).__init__(testset, net, scaler_flux, scaler_cont, smooth=smooth,
                                                  interpolate=interpolate)

    def random_index(self, size=1):
        '''Draw size random indices in order to plot random predictions.'''

        rand_indx = np.random.randint(0, len(self.flux), size)

        return rand_indx


    def create_figure(self, figsize=(7,5), dpi=320):
        '''Create a figure to add subplots to.'''

        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.axes = []

        return self.fig


    def plot(self, index, figsize=(7,5), dpi=320, subplotloc=111,\
             alpha=0.7, contpredcolor="darkred", includesmooth=True,\
             fluxsmoothcolor="navy", drawsplit=True, wave_split=1216,
             wave_min=1020., wave_max=1970.):
        '''Plot the prediction for the spectrum of a certain index.'''

        cont_pred = self.cont_pred_np[index].squeeze()

        try:
            fig = self.fig
        except:
            fig = self.create_figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(subplotloc)

        # squeeze only works if there is no noise channel
        try:
            ax.plot(self.wave_grid, self.flux[index].squeeze(), alpha=alpha, lw=1, \
                    label="Mock spectrum")
        except:
            ax.plot(self.wave_grid, self.flux[index], alpha=alpha, lw=1,
                    label="Mock spectrum")

        if self.noise is not None:
            ax.plot(self.wave_grid, self.noise[index].squeeze(), alpha=alpha, lw=.5,
                    label="Noise", c="green")

        if self.cont is not None:
            ax.plot(self.wave_grid, self.cont[index].squeeze(), alpha=alpha, lw=2, \
                    label="True continuum")
        ax.plot(self.wave_grid, cont_pred, alpha=alpha, lw=1, ls="--",\
                label="Predicted continuum", color=contpredcolor)
        if includesmooth:
            try:
                flux_smooth = self.flux_smooth
                ax.plot(self.wave_grid, flux_smooth[index].squeeze(), alpha=alpha, lw=1,\
                        ls="dashdot", label="Smoothed spectrum",\
                        color=fluxsmoothcolor)
            except:
                print ("Warning: flux has not been smoothed.")

        if drawsplit:
            ax.axvline(wave_split, alpha=0.7, lw=1., ls="dashdot", color="black", label="Blue-red split")

        ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"$F / F_{1280 \AA}$")
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)

        ax.set_xlim(wave_min, wave_max)

        ax.set_title("Results for test spectrum "+str(index+1))

        self.axes.append(ax)

        return ax


    def plot_scaled(self, index, figsize=(7,5), dpi=320, subplotloc=111, alpha=0.7,\
                    contpredcolor="darkred", drawsplit=True, wave_split=1216):

        if not self.use_QSOScaler:
            print ("Warning: no scaling involved!")
            return

        cont_pred_scaled = self.cont_pred_scaled_np[index].squeeze()

        try:
            fig = self.fig
        except:
            fig = self.create_figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(subplotloc)

        try:
            ax.plot(self.wave_grid, self.flux_scaled[index].squeeze(), alpha=alpha, lw=1,\
                    label="Mock spectrum", c="tab:blue")
        except:
            ax.plot(self.wave_grid, self.flux_scaled[index,0], alpha=alpha, lw=1,
                    label="Mock spectrum", c="tab:blue")

        ax.plot(self.wave_grid, self.cont_true_scaled_np[index].squeeze(), alpha=alpha, lw=2,\
                label="True continuum", c="tab:orange")
        ax.plot(self.wave_grid, cont_pred_scaled, alpha=alpha, lw=1, ls="--",\
                label="Predicted continuum", color=contpredcolor)

        if drawsplit:
            ax.axvline(wave_split, alpha=0.7, lw=1., ls="dashdot", color="black", label="Blue-red split")

        ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
        ax.set_ylabel("Scaled flux")
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        ax.grid(which="minor", linewidth=.1, alpha=.3, color="grey")
        ax.set_title("Raw network output for test spectrum "+str(index+1))


    def plot_scaled_pixels(self, index, figsize=(7,5), dpi=320, subplotloc=111, alpha=0.7,\
                           contpredcolor="darkred"):

        pixels = np.arange(1, len(self.wave_grid)+1, 1)

        if not self.use_QSOScaler:
            print ("Warning: no scaling involved!")
            return

        cont_pred_scaled = self.cont_pred_scaled_np[index].squeeze()

        try:
            fig = self.fig
        except:
            fig = self.create_figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(subplotloc)

        try:
            ax.plot(pixels, self.flux_scaled[index].squeeze(), alpha=alpha, lw=1,\
                    label="Mock spectrum", c="tab:blue")
        except:
            ax.plot(pixels, self.flux_scaled[index,0], alpha=alpha, lw=1,
                    label="Mock spectrum", c="tab:blue")

        ax.plot(pixels, self.cont_true_scaled_np[index].squeeze(), alpha=alpha, lw=2,\
                label="True continuum", c="tab:orange")
        ax.plot(pixels, cont_pred_scaled, alpha=alpha, lw=1, ls="--",\
                label="Predicted continuum", color=contpredcolor)

        ax.set_xlabel("Pixel number")
        ax.set_ylabel("Scaled flux")
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        ax.grid(which="minor", linewidth=.1, alpha=.3, color="grey")
        ax.set_title("Raw network output for test spectrum "+str(index+1))


    def show_figure(self):

        self.fig.tight_layout()
        self.fig.show()


class RelResids(ModelResults):
    def __init__(self, testset, net, scaler_flux, scaler_cont, smooth=False, interpolate=False):
        super(RelResids, self).__init__(testset, net, scaler_flux, scaler_cont, smooth=smooth, interpolate=interpolate)

        print ("Starting initialisation of the RelResids instance.")
        print ("self.uni_cont.shape:", self.uni_cont.shape)
        print ("self.uni_cont_pred.shape:", self.uni_cont_pred.shape)

        if interpolate:
            # overwrite the hybrid grid with the uniform grid for simpler plotting code
            rel_resid = (self.uni_cont - self.uni_cont_pred) / self.uni_cont
            print ("Extracted residuals.")
            self.wave_grid = self.uni_wave_grid
            print ("Extracted uniform wavelength grid.")

        else:
            rel_resid = (self.cont - self.cont_pred_np) / self.cont
            rel_resid = rel_resid.cpu().detach().numpy()

        self.rel_resid = rel_resid.squeeze()
        self.mean_spec = np.mean(self.rel_resid, axis=0)
        self.std_spec = np.std(self.rel_resid, axis=0)
        self.mad_std_spec = mad_std(self.rel_resid, axis=0)

        self.mean_resid = np.mean(self.rel_resid)
        self.std_resid = np.std(self.rel_resid)
        self.mad_resid = mad_std(self.rel_resid)

        percent_min_1sig = np.percentile(self.rel_resid, 100. * norm.cdf(-1.), axis=0)
        percent_plu_1sig = np.percentile(self.rel_resid, 100. * norm.cdf(1.), axis=0)
        self.percentile_std = (percent_plu_1sig - percent_min_1sig) / 2.
        self.percentile_median = np.percentile(self.rel_resid, 50., axis=0)

        self.sigma_min = percent_min_1sig
        self.sigma_plu = percent_plu_1sig

        print ("Computed summary statistics.")


class ScaledResids(ModelResults):
    def __init__(self, testset, net, scaler_flux, scaler_cont, smooth=False):
        super(ScaledResids, self).__init__(testset, net, scaler_flux, scaler_cont, smooth=smooth)

        scaled_resid = (self.cont_true_scaled_np - self.cont_pred_scaled_np) / self.cont_true_scaled_np
        self.scaled_resid = scaled_resid.squeeze()
        self.mean_spec = np.mean(self.scaled_resid, axis=0)
        self.std_spec = np.std(self.scaled_resid, axis=0)
        self.mad_std_spec = mad_std(self.scaled_resid, axis=0)

        self.mean_resid = np.mean(self.scaled_resid)
        self.std_resid = np.std(self.scaled_resid)
        self.mad_resid = mad_std(self.scaled_resid)


class CorrelationMatrix(RelResids):
    def __init__(self, testset, net, scaler_flux, scaler_cont, smooth=False):
        super(CorrelationMatrix, self).__init__(testset, net, scaler_flux, scaler_cont, smooth=smooth)

        diff = self.rel_resid - self.mean_spec
        covar_delta = 1 / (self.cont.shape[0] - 1) * np.matmul(diff.T, diff)
        corr_delta = covar_delta / np.sqrt(np.outer(np.diag(covar_delta), np.diag(covar_delta)))
        self.matrix = corr_delta

    def show(self):
        '''Show the correlation matrix on a wavelength-wavelength grid.'''

        self.fig, self.ax = plt.subplots(figsize=(7,5), dpi=320)
        self.im = self.ax.pcolormesh(self.wave_grid, self.wave_grid, self.matrix, cmap="bwr",\
                                     shading="nearest", vmin=-1.0, vmax=1.0)
        self.ax.set_aspect("equal")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="Residual correlation")
        self.ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_ylabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_title("Correlation matrix of residuals")

        self.ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        self.fig.show()

        return self.fig, self.ax

    def savefig(self, fullfilename):
        '''Save the figure to a file with full path fullfilename. Only works if the figure is already created.'''

        try:
            self.fig.savefig(fullfilename)
        except:
            print ("Could not save a figure. Does it exist yet?")


class ResidualPlots(RelResids):
    def __init__(self, testset, net, scaler_flux, scaler_cont, smooth=False, interpolate=False):

        super(ResidualPlots, self).__init__(testset, net, scaler_flux, scaler_cont, smooth=smooth,
                                            interpolate=interpolate)


    def plot_means(self, show_std=False, drawsplit=True, wave_split=1216,
                   wave_min=1020., wave_max=1970., figsize=(6,4), dpi=320):
        '''Plot the mean relative residuals as a function of wavelength, and add the deviations as shaded areas.'''

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(self.wave_grid, self.mean_spec, label="Mean", color="black")
        if show_std:
            ax.fill_between(self.wave_grid, self.mean_spec-self.std_spec, self.mean_spec+self.std_spec, alpha=0.3,\
                            label="Standard deviation", color="tab:blue")
        ax.fill_between(self.wave_grid, self.mean_spec-self.mad_std_spec, self.mean_spec+self.mad_std_spec, alpha=0.3,\
                        label="MAD standard deviation", color="tab:orange")

        if drawsplit:
            ax.axvline(wave_split, alpha=0.7, lw=2, ls="dashdot", color="black", label="Blue-red split")

        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        ax.grid(which="minor", linewidth=.1, alpha=.3, color="grey")
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel("$\\frac{F_{true} - F_{pred}}{F_{true}}$")

        ax.set_xlim(wave_min, wave_max)

        ax.set_title("Residuals relative to true continuum")

        return fig, ax


    def plot_percentiles(self, wave_min=1020., wave_max=1970., figsize=(6,4), dpi=320):

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.plot(self.wave_grid, self.percentile_median, label="Median", color="black")
        ax.fill_between(self.wave_grid, self.sigma_min, self.sigma_plu, alpha=0.3, \
                        label=r"68\% interval", color="tab:orange")

        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"$(F_\textrm{true} - F_\textrm{pred}) / F_\textrm{true}$")
        ax.set_title("Residuals Relative to True Continuum")

        ax.set_xlim(wave_min, wave_max)

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


class ScaledResidualPlots(ScaledResids):
    def __init__(self, testset, net, scaler_flux, scaler_cont, smooth=False):

        super(ScaledResidualPlots, self).__init__(testset, net, scaler_flux, scaler_cont, smooth=smooth)


    def plot_means(self, show_std=False):
        '''Plot the mean relative residuals as a function of wavelength, and add the deviations as shaded areas.'''

        fig, ax = plt.subplots(figsize=(7,5), dpi=320)
        ax.plot(self.wave_grid, self.mean_spec, label="Mean", color="black", lw=1, ls="--")
        if show_std:
            ax.fill_between(self.wave_grid, self.mean_spec-self.std_spec, self.mean_spec+self.std_spec, alpha=0.3,\
                            label="Standard deviation", color="tab:blue")
        ax.fill_between(self.wave_grid, self.mean_spec-self.mad_std_spec, self.mean_spec+self.mad_std_spec, alpha=0.3,\
                        label="MAD standard deviation", color="tab:orange")
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major")
        ax.grid(which="minor", linewidth=.1, alpha=.3, color="grey")
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel("$\\frac{f_{true} - f_{pred}}{f_{true}}$")
        ax.set_title("Scaled residuals relative to scaled true continuum")

        return fig, ax

    def resid_hist(self):

        fig, ax = plt.subplots()
        hist, bins, patches = ax.hist(self.scaled_resid.flatten(), bins=100, density=True, range=(-0.3, 0.3),\
                                       label="mean={:5.3f}, std = {:5.3f}, 1.48*mad={:5.3f}".format(self.mean_resid,\
                                                                                                    self.std_resid,\
                                                                                                    self.mad_resid))
        bin_cen = (bins[:-1] + bins[1:])/2
        ax.plot(bin_cen, norm.pdf(bin_cen, loc=self.mean_resid, scale=self.mad_resid), label="Gaussian (with MAD std)")
        ax.set_xlabel("$\\frac{f_{true} - f_{pred}}{f_{true}}$")
        ax.set_ylabel("Probability density")
        ax.set_title("Scaled residuals relative to scaled true continuum")
        ax.legend()
        return fig, ax