from learning.testing import ModelResults, CorrelationMatrix, ResidualStatistics
from utils.smooth_scaler import SmoothScaler
import torch
import numpy as np
from astropy.stats import mad_std
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils.errorfuncs import corr_matrix_relresids


class DoubleScalingResults:
    '''Class for predicting the continua for the test set and converting
    everything to numpy arrays. Meant to be used as a parent class for
    other testing classes.'''

    def __init__(self, testset, net, globscaler_flux, globscaler_cont):
        self.wave_grid = testset.wave_grid
        self.flux = testset.flux
        self.cont = testset.cont
        self.flux_smooth = testset.flux_smooth
        self.globscaler_flux = globscaler_flux
        self.globscaler_cont = globscaler_cont
        self.net = net

        # get all predictions on the test set
        # set up the local scaler
        loc_scaler = SmoothScaler(self.wave_grid, self.flux_smooth)
        self.loc_scaler = loc_scaler

        # doubly transform the input flux
        flux_scaled = loc_scaler.forward(torch.FloatTensor(self.flux))
        flux_scaled = self.globscaler_flux.forward(flux_scaled)

        # forward the network
        cont_pred_scaled = self.net.forward(flux_scaled)

        # detransform
        cont_pred_descaled = self.globscaler_cont.backward(cont_pred_scaled)
        cont_pred = loc_scaler.backward(cont_pred_descaled)

        self.cont_pred = cont_pred.detach().numpy()


class DoubleScalingResultsSpectra(DoubleScalingResults):
    '''Class for plotting model output on test spectra.'''

    def __init__(self, testset, net, globscaler_flux, globscaler_cont):
        super(DoubleScalingResultsSpectra, self).__init__(testset, net, globscaler_flux,\
                                                   globscaler_cont)


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
             fluxsmoothcolor="navy"):
        '''Plot the prediction for the spectrum of a certain index.'''

        cont_pred = self.cont_pred[index]

        try:
            fig = self.fig
        except:
            fig = self.create_figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(subplotloc)

        ax.plot(self.wave_grid, self.flux[index], alpha=alpha, lw=1,\
                label="Mock spectrum")
        ax.plot(self.wave_grid, self.cont[index], alpha=alpha, lw=2,\
                label="True continuum")
        ax.plot(self.wave_grid, cont_pred, alpha=alpha, lw=1, ls="--",\
                label="Predicted continuum", color=contpredcolor)
        if includesmooth:
            try:
                flux_smooth = self.flux_smooth
                ax.plot(self.wave_grid, flux_smooth[index], alpha=alpha, lw=1,\
                        ls="dashdot", label="Smoothed spectrum",\
                        color=fluxsmoothcolor)
            except:
                print ("Warning: flux has not been smoothed.")

        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel("Normalised flux")
        ax.legend()
        ax.grid()
        ax.set_title("Results for test spectrum "+str(index+1))

        self.axes.append(ax)

        return ax


    def show_figure(self):

        self.fig.tight_layout()
        self.fig.show()


class DoubleScalingCorrelationMatrix(DoubleScalingResults):
    '''Class for plotting the correlation matrix.'''

    def __init__(self, testset, net, globscaler_flux, globscaler_cont):
        super(DoubleScalingCorrelationMatrix, self).__init__(testset,\
                                                             net,\
                                                             globscaler_flux,\
                                                             globscaler_cont)

        rel_resid = (self.cont - self.cont_pred) / self.cont
        mean_spec = np.mean(rel_resid, axis=0)
        diff = rel_resid - mean_spec
        covar_delta = 1/(self.cont.shape[0]-1) * np.matmul(diff.T, diff)
        corr_delta = covar_delta/np.sqrt(np.outer(np.diag(covar_delta), np.diag(covar_delta)))
        self.matrix = corr_delta


    def show(self):
        '''Show the correlation matrix on a wavelength-wavelength grid.'''

        self.fig, self.ax = plt.subplots(figsize=(7,5), dpi=320)
        self.im = self.ax.pcolormesh(self.wave_grid, self.wave_grid, self.matrix, cmap="bwr", shading="nearest",\
                                     vmin=-1.0, vmax=1.0)
        self.ax.set_aspect("equal")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="Residual correlation")
        self.ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_ylabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_title("Correlation matrix of residuals")
        self.fig.show()

        return self.fig, self.ax


class DoubleScalingResidStats(DoubleScalingResults):
    '''Class for making plots of the relative residuals.'''

    def __init__(self, testnet, net, globscaler_flux, globscaler_cont):
        super(DoubleScalingResidStats, self).__init__(testnet, net,\
                                                      globscaler_flux,\
                                                      globscaler_cont)

        self.rel_resid = (self.cont - self.cont_pred) / self.cont
        self.mean_spec = np.mean(self.rel_resid, axis=0)
        self.std_spec = np.std(self.rel_resid, axis=0)
        self.mad_std_spec = mad_std(self.rel_resid, axis=0)

        self.mean_resid = np.mean(self.rel_resid)
        self.std_resid = np.std(self.rel_resid)
        self.mad_resid = mad_std(self.rel_resid)


    def resid_hist(self):
        fig, ax = plt.subplots()
        hist, bins, patches = ax.hist(self.rel_resid.flatten(), bins=100, density=True, range=(-0.3, 0.3), \
                                      label="mean={:5.3f}, std = {:5.3f}, 1.48*mad={:5.3f}".format(self.mean_resid, \
                                                                                                   self.std_resid, \
                                                                                                   self.mad_resid))
        bin_cen = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_cen, norm.pdf(bin_cen, loc=self.mean_resid, scale=self.mad_resid), label="Gaussian (with MAD std)")
        ax.set_xlabel("Relative residual")
        ax.set_ylabel("Probability density")
        ax.set_title("Residuals relative to input flux")
        ax.legend()
        return fig, ax

    def plot_means(self, show_std=False):
        '''Plot the mean relative residuals as a function of wavelength, and add the deviations as shaded areas.'''

        fig, ax = plt.subplots(figsize=(7, 5), dpi=320)
        ax.plot(self.wave_grid, self.mean_spec, label="Mean", color="black")
        if show_std:
            ax.fill_between(self.wave_grid, self.mean_spec - self.std_spec, self.mean_spec + self.std_spec, alpha=0.3, \
                            label="Standard deviation", color="tab:blue")
        ax.fill_between(self.wave_grid, self.mean_spec - self.mad_std_spec, self.mean_spec + self.mad_std_spec, alpha=0.3, \
                        label="MAD standard deviation", color="tab:orange")
        ax.legend()
        ax.grid()
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel("Relative residual")
        ax.set_title("Residuals relative to input flux")

        return fig, ax

# -------------------------------------
# deprecated classes

class ModelResultsDoublyScaled(ModelResults):
    def __init__(self, testset, net, globscaler_flux, globscaler_cont):
        wave_grid = testset.wave_grid
        flux = testset.flux
        cont = testset.cont
        self.flux_smooth = testset.flux_smooth

        super(ModelResultsDoublyScaled, self).__init__(wave_grid, flux,\
                                                       cont, net,\
                                                       globscaler_flux,\
                                                       globscaler_cont)

    def predict_numpy(self):
        '''Gets all the predictions on the test set and converts everything
        to numpy arrays.'''

        # set up the local scaler
        loc_scaler = SmoothScaler(self.wave_grid, self.flux_smooth)

        # doubly transform the input flux
        flux_scaled = loc_scaler.forward(torch.FloatTensor(self.flux_test))
        flux_scaled = self.scaler_flux.forward(flux_scaled)

        # forward the network
        cont_pred_scaled = self.net.forward(flux_scaled)

        # detransform
        cont_pred_descaled = self.scaler_cont.backward(cont_pred_scaled)
        cont_pred = loc_scaler.backward(cont_pred_descaled)

        return cont_pred.detach().numpy()


class ResidualStatisticsDoublyScaled:
    def __init__(self, testset, net, globscaler_flux, globscaler_cont):
        self.wave_grid = testset.wave_grid
        self.flux = testset.flux
        self.cont = testset.cont
        self.flux_smooth = testset.flux_smooth
        self.globscaler_flux = globscaler_flux
        self.globscaler_cont = globscaler_cont
        self.net = net

    def _predict_numpy(self):
        '''Gets all the predictions on the test set and converts everything
        to numpy arrays.'''

        # set up the local scaler
        loc_scaler = SmoothScaler(self.wave_grid, self.flux_smooth)

        # doubly transform the input flux
        flux_scaled = loc_scaler.forward(torch.FloatTensor(self.flux))
        flux_scaled = self.globscaler_flux.forward(flux_scaled)

        # forward the network
        cont_pred_scaled = self.net.forward(flux_scaled)

        # detransform
        cont_pred_descaled = self.globscaler_cont.backward(cont_pred_scaled)
        cont_pred = loc_scaler.backward(cont_pred_descaled)

        return cont_pred.detach().numpy()

    def compute_stats(self):

        cont_pred = self._predict_numpy()

        self.rel_resid = (self.cont - cont_pred) / self.cont
        self.mean_spec = np.mean(self.rel_resid, axis=0)
        self.std_spec = np.std(self.rel_resid, axis=0)
        self.mad_std_spec = mad_std(self.rel_resid, axis=0)

        self.mean_resid = np.mean(self.rel_resid)
        self.std_resid = np.std(self.rel_resid)
        self.mad_resid = mad_std(self.rel_resid)

    def resid_hist(self):
        fig, ax = plt.subplots()
        hist, bins, patches = ax.hist(self.rel_resid.flatten(), bins=100, density=True, range=(-0.3, 0.3), \
                                      label="mean={:5.3f}, std = {:5.3f}, 1.48*mad={:5.3f}".format(self.mean_resid, \
                                                                                                   self.std_resid, \
                                                                                                   self.mad_resid))
        bin_cen = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_cen, norm.pdf(bin_cen, loc=self.mean_resid, scale=self.mad_resid), label="Gaussian (with MAD std)")
        ax.set_xlabel("Relative residual")
        ax.set_ylabel("Probability density")
        ax.set_title("Residuals relative to input flux")
        ax.legend()
        return fig, ax

    def plot_means(self, show_std=False):
        '''Plot the mean relative residuals as a function of wavelength, and add the deviations as shaded areas.'''

        fig, ax = plt.subplots(figsize=(7, 5), dpi=320)
        ax.plot(self.wave_grid, self.mean_spec, label="Mean", color="black")
        if show_std:
            ax.fill_between(self.wave_grid, self.mean_spec - self.std_spec, self.mean_spec + self.std_spec, alpha=0.3, \
                            label="Standard deviation", color="tab:blue")
        ax.fill_between(self.wave_grid, self.mean_spec - self.mad_std_spec, self.mean_spec + self.mad_std_spec, alpha=0.3, \
                        label="MAD standard deviation", color="tab:orange")
        ax.legend()
        ax.grid()
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel("Relative residual")
        ax.set_title("Residuals relative to input flux")

        return fig, ax