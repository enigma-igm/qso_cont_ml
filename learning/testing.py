import numpy as np
import matplotlib.pyplot as plt
from utils.errorfuncs import relative_residuals, corr_matrix_relresids

class CorrelationMatrix:
    def __init__(self, flux_test, cont_test, scaler_flux, scaler_cont, net):
        self.X_test = flux_test
        self.y_test = cont_test
        self.scaler_X = scaler_flux
        self.scaler_y = scaler_cont
        self.net = net

        # compute the correlation matrix
        # first forward the model
        result_test = net.full_predict(flux_test, self.scaler_X, self.scaler_y)
        self.matrix = corr_matrix_relresids(self.y_test, result_test, len(self.y_test))

    def show(self, wave_grid):
        '''Show the correlation matrix on a wavelength-wavelength grid.'''

        self.fig, self.ax = plt.subplots(figsize=(7,5), dpi=320)
        self.im = self.ax.pcolormesh(wave_grid, wave_grid, self.matrix)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="Correlation")
        self.ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_ylabel("Rest-frame wavelength ($\AA$)")
        self.ax.set_title("Correlation matrix")
        self.fig.show()

        return self.fig, self.ax

    def savefig(self, fullfilename):
        '''Save the figure to a file with full path fullfilename. Only works if the figure is already created.'''

        try:
            self.fig.savefig(fullfilename)
        except:
            print ("Could not save a figure. Does it exist yet?")


class ResidualStatistics:
    def __init__(self, flux_test, cont_test, scaler_flux, scaler_cont, net):
        self.X_test = flux_test
        self.y_test = cont_test
        self.scaler_X = scaler_flux
        self.scaler_y = scaler_cont
        self.net = net

        # compute residuals relative to flux in absorption spectrum (for every point on the wavelength grid)
        # compute some statistics
        # first forward the model to get predictions
        result_test = net.full_predict(self.X_test, self.scaler_X, self.scaler_y)

        # compute stats
        self.rel_resid, self.mean_spec, self.std_spec, self.mad_std_spec = relative_residuals(self.y_test, result_test)

    def plot_means(self, wave_grid):
        '''Plot the mean relative residuals as a function of wavelength, and add the deviations as shaded areas.'''

        fig, ax = plt.subplots(figsize=(7,5), dpi=320)
        ax.plot(wave_grid, self.mean_spec, label="Mean")
        ax.fill_between(wave_grid, self.mean_spec-self.std_spec, self.mean_spec+self.std_spec, alpha=0.3,\
                        label="Standard deviation")
        ax.fill_between(wave_grid, self.mean_spec-self.mad_std_spec, self.mean_spec+self.mad_std_spec, alpha=0.3,\
                        label="MAD standard deviation")
        ax.legend()
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel("Relative residual")
        ax.set_title("Residuals relative to input flux")

        return fig, ax