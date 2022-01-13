import numpy as np
import matplotlib.pyplot as plt
from utils.errorfuncs import relative_residuals, corr_matrix_relresids
from astropy.stats import mad_std
from scipy.stats import norm
import torch
from pypeit.utils import fast_running_median


class ModelResults:
    '''Class for easily plotting model output on test spectra.'''

    def __init__(self, wave_grid, flux_test, cont_test, net, scaler_flux=None,\
                 scaler_cont=None, smooth=False):
        self.wave_grid = wave_grid
        self.flux_test = flux_test
        self.cont_test = cont_test
        self.scaler_flux = scaler_flux
        self.scaler_cont = scaler_cont
        self.net = net
        self.smooth = smooth

        if scaler_flux is None:
            self.use_QSOScaler = False
        else:
            self.use_QSOScaler = True


    def predict_numpy(self):
        '''Gets all the predictions on the test set and converts everything to
        numpy arrays.'''

        flux_tensor = torch.FloatTensor(self.flux_test)

        # smooth input for the final skip connection before applying the QSOScaler
        if self.smooth:
            input_smooth = np.zeros(self.flux_test.shape)
            for i in range(len(input_smooth)):
                input_smooth[i] = fast_running_median(self.flux_test[i], 20)
            input_smooth = torch.FloatTensor(input_smooth)
            self.flux_smooth = input_smooth.detach().numpy()

        else:
            input_smooth = None

        if self.use_QSOScaler:
            input = self.scaler_flux.forward(flux_tensor)

            if self.smooth:
                input_smooth = self.scaler_flux.forward(input_smooth)

            res = self.net(input, smooth=self.smooth, x_smooth=input_smooth)
            res_descaled = self.scaler_cont.backward(res)
            res_np = res_descaled.detach().numpy()

        else:
            res = self.net(flux_tensor, smooth=self.smooth, x_smooth=input_smooth)
            res_np = res.detach().numpy()

        self.cont_pred_np = res_np
        return self.cont_pred_np


    def random_index(self, size=1):
        '''Draw size random indices in order to plot random predictions.'''

        rand_indx = np.random.randint(0, len(self.flux_test), size)

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

        try:
            preds = self.cont_pred_np
        except:
            preds = self.predict_numpy()

        cont_pred = preds[index]

        try:
            fig = self.fig
        except:
            fig = self.create_figure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(subplotloc)

        ax.plot(self.wave_grid, self.flux_test[index], alpha=alpha, lw=1,\
                label="Mock spectrum")
        ax.plot(self.wave_grid, self.cont_test[index], alpha=alpha, lw=2,\
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



class CorrelationMatrix:
    def __init__(self, flux_test, cont_test, scaler_flux, scaler_cont, net,\
                 smooth=False):
        self.X_test = flux_test
        self.y_test = cont_test
        self.scaler_X = scaler_flux
        self.scaler_y = scaler_cont
        self.net = net

        if scaler_flux is None:
            self.use_QSOScaler = False
        else:
            self.use_QSOScaler = True

        if smooth:
            x_smooth = np.zeros(flux_test.shape)
            for i in range(len(flux_test)):
                x_smooth[i] = fast_running_median(flux_test[i], 20)

        else:
            x_smooth = None

        # compute the correlation matrix
        # first forward the model
        result_test = net.full_predict(flux_test, scaler_flux, scaler_cont,\
                                       smooth=smooth, x_smooth=x_smooth)
        #if self.use_QSOScaler:
        #    result_test = net.full_predict(flux_test, self.scaler_X, self.scaler_y)
        #else:
        #    result_test = net.forward(flux_test)

        self.matrix = corr_matrix_relresids(self.y_test, result_test, len(self.y_test))

    def show(self, wave_grid):
        '''Show the correlation matrix on a wavelength-wavelength grid.'''

        self.fig, self.ax = plt.subplots(figsize=(7,5), dpi=320)
        self.im = self.ax.pcolormesh(wave_grid, wave_grid, self.matrix, cmap="bwr", shading="nearest",\
                                     vmin=-1.0, vmax=1.0)
        self.ax.set_aspect("equal")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="Residual correlation")
        self.ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_ylabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_title("Correlation matrix of residuals")
        self.fig.show()

        return self.fig, self.ax

    def savefig(self, fullfilename):
        '''Save the figure to a file with full path fullfilename. Only works if the figure is already created.'''

        try:
            self.fig.savefig(fullfilename)
        except:
            print ("Could not save a figure. Does it exist yet?")


class ResidualStatistics:
    def __init__(self, flux_test, cont_test, scaler_flux, scaler_cont, net,\
                 smooth=False):
        self.X_test = flux_test
        self.y_test = cont_test
        self.scaler_X = scaler_flux
        self.scaler_y = scaler_cont
        self.net = net

        if scaler_flux is None:
            self.use_QSOScaler = False
        else:
            self.use_QSOScaler = True

        if smooth:
            x_smooth = np.zeros(flux_test.shape)
            for i in range(len(flux_test)):
                x_smooth[i] = fast_running_median(flux_test[i], 20)

        else:
            x_smooth = None

        # compute residuals relative to flux in absorption spectrum (for every point on the wavelength grid)
        # compute some statistics
        # first forward the model to get predictions
        result_test = net.full_predict(flux_test, scaler_flux, scaler_cont,\
                                       smooth=smooth, x_smooth=x_smooth)
        #if self.use_QSOScaler:
        #    result_test = net.full_predict(self.X_test, self.scaler_X, self.scaler_y)
        #else:
        #    result_test = net.forward(self.X_test)

        # compute stats
        self.rel_resid = (self.y_test - result_test)/self.y_test
        self.mean_spec = np.mean(self.rel_resid, axis=0)
        self.std_spec = np.std(self.rel_resid, axis=0)
        self.mad_std_spec = mad_std(self.rel_resid, axis=0)

        self.mean_resid = np.mean(self.rel_resid)
        self.std_resid = np.std(self.rel_resid)
        self.mad_resid = mad_std(self.rel_resid)
        #self.rel_resid, self.mean_spec, self.std_spec, self.mad_std_spec = relative_residuals(self.y_test, result_test)

    def plot_means(self, wave_grid, show_std=False):
        '''Plot the mean relative residuals as a function of wavelength, and add the deviations as shaded areas.'''

        fig, ax = plt.subplots(figsize=(7,5), dpi=320)
        ax.plot(wave_grid, self.mean_spec, label="Mean", color="black")
        if show_std:
            ax.fill_between(wave_grid, self.mean_spec-self.std_spec, self.mean_spec+self.std_spec, alpha=0.3,\
                            label="Standard deviation", color="tab:blue")
        ax.fill_between(wave_grid, self.mean_spec-self.mad_std_spec, self.mean_spec+self.mad_std_spec, alpha=0.3,\
                        label="MAD standard deviation", color="tab:orange")
        ax.legend()
        ax.grid()
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        #ax.set_ylabel("Relative residual")
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
        #ax.set_xlabel("Relative residual")
        ax.set_xlabel("$\\frac{F_{true} - F_{pred}}{F_{true}}$")
        ax.set_ylabel("Probability density")
        ax.set_title("Residuals relative to true continuum")
        ax.legend()
        return fig, ax