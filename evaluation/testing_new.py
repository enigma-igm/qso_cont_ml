import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.stats import norm
from scipy.interpolate import interp1d   # necessary?
from data.load_data_new import SynthSpectra
#from data.wavegrid_conversion import InputSpectra
#from qso_fitting.data.sdss.sdss import autofit_continua, qsmooth_continua


class ModelResults:
    '''
    Class for computing network predictions for a test set. The user chooses onto which grid the network output is
    stored: coarse, fine or hybrid.

    Currently only supports synthetic test spectra.

    TO DO: enable other InputSpectra.

    Attributes:
        device: torch.device
        input_hybrid: torch tensor of shape (n_qso, 3, n_wav)
        flux: ndarray of shape (n_qso, n_wav)
        noise: ndarray of shape (n_qso, n_wav)
        cont_pred: ndarray of shape (n_qso, n_wav)
        cont_true: ndarray of shape (n_qso, n_wav)
        wave_grid: ndarray of shape (n_wav,)
        grid_type: str
    '''

    def __init__(self, testset, net, scaler_hybrid, gridtype="hybrid"):

        if not isinstance(testset, SynthSpectra):
            raise TypeError("'testset' must be a SynthSpectra instance.")
        # TO DO: also incorporate possibility of providing prepped empirical spectra

        # update the device of the scalers if necessary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scaler_hybrid.updateDevice()

        self.input_hybrid = testset.input_hybrid

        # scale the 3D input tensor
        input_scaled = scaler_hybrid.forward(self.input_hybrid)

        # apply the network to the input and descale the predictions again
        # squeeze the predictions such that the feature channel dimension is gone
        res_scaled = net(input_scaled)
        res_descaled = scaler_hybrid.backward(res_scaled).squeeze(dim=1)

        # make numpy arrays for easier plotting and manipulation
        res_hybrid_np = res_descaled.cpu().detach().numpy()

        # put everything onto the desired grid and save as numpy arrays
        # also load the true continua on the same grid
        # as well as the absorption spectra and noise

        # TO DO: enable NoneType true continuum for real test spectra
        if gridtype == "coarse":
            self.cont_pred = interp1d(testset.wave_hybrid, res_hybrid_np, kind="cubic", axis=-1, bounds_error=False,
                                      fill_value="extrapolate")(testset.wave_coarse)
            self.cont_true = testset.cont_coarse.cpu().detach().numpy()
            self.flux = testset.flux_coarse.cpu().detach().numpy()
            self.noise = testset.noise_coarse.cpu().detach().numpy()
            self.wave_grid = testset.wave_coarse

        elif gridtype == "fine":
            self.cont_pred = interp1d(testset.wave_hybrid, res_hybrid_np, kind="cubic", axis=-1, bounds_error=False,
                                      fill_value="extrapolate")(testset.wave_fine)
            self.cont_true = testset.cont_fine.cpu().detach().numpy()
            self.flux = testset.flux_fine.cpu().detach().numpy()
            self.noise = testset.noise_fine.cpu().detach().numpy()
            self.wave_grid = testset.wave_fine

        elif gridtype == "hybrid":
            self.cont_pred = res_hybrid_np
            self.cont_pred = testset.cont_hybrid.cpu().detach().numpy()
            self.flux = testset.flux_hybrid.cpu().detach().numpy()
            self.noise = 1 / np.sqrt(testset.ivar_hybrid.cpu().detach().numpy())
            self.wave_grid = testset.wave_hybrid

        else:
            raise ValueError("Parameter 'gridtype' must be 'coarse', 'fine' or 'hybrid'.")

        self.grid_type = gridtype


class ModelResultsSpectra(ModelResults):

    def __init__(self, testset, net, scaler_hybrid, gridtype="coarse"):

        super(ModelResultsSpectra, self).__init__(testset, net, scaler_hybrid, gridtype)


    def randomIndex(self, size=1):
        '''
        Draw random indices of quasars in the sample.

        @param size:
        @return:
        '''

        rand_indx = np.random.randint(0, len(self.cont_pred), size)

        return rand_indx


    def createFigure(self, figsize=(6,4), dpi=320):
        '''
        Create a figure to add subplots to.

        @param figsize:
        @param dpi:
        @return:
        '''

        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.axes = []

        return self.fig


    def plot(self, index, figsize=(6,4), dpi=320, subplotloc=111, alpha=0.7, alpha_pred=0.9, contpredcolor="darkred",
             wave_min=1020., wave_max=1970.):
        '''
        Plot the predicted continuum of the test spectrum with a given index.

        @param index:
        @param figsize:
        @param dpi:
        @param subplotloc:
        @param alpha:
        @param alpha_pred:
        @param contpredcolor:
        @param wave_min:
        @param wave_max:
        @return:
        '''

        try:
            fig = self.fig
        except:
            fig = self.createFigure(figsize=figsize, dpi=dpi)

        ax = fig.add_subplot(subplotloc)

        ax.plot(self.wave_grid, np.squeeze(self.flux[index]), alpha=alpha, lw=1., label="Mock spectrum", c="tab:blue")
        ax.plot(self.wave_grid, np.squeeze(self.noise[index]), alpha=alpha, lw=.5, label="Noise", c="green")

        if self.cont_true is not None:
            ax.plot(self.wave_grid, np.squeeze(self.cont_true[index]), alpha=alpha, lw=2., label="True continuum", c="tab:orange")

        ax.plot(self.wave_grid, np.squeeze(self.cont_pred[index]), alpha=alpha_pred, lw=1.5, ls="--", c=contpredcolor,
                label="Predicted continuum")

        ax.axvline(1216., alpha=0.7, lw=1., ls="dashdot", color="black", label="Blue-red split")

        ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"$F / F_{1280 \AA}$")
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)

        ax.set_xlim(wave_min, wave_max)

        ax.set_title("Results for test spectrum " + str(index + 1))

        self.axes.append(ax)

        return ax


    def showFigure(self):
        '''
        Show the figure.

        @return:
        '''

        self.fig.tight_layout()
        self.fig.show()


class RelResids(ModelResults):
    '''
    Computes the residuals relative to the true continuum on the desired grid, and extracts the median and 68%
    intervals.

    The residuals are defined as: Delta / F_true = (F_true - F_pred) / F_true.
    '''

    def __init__(self, testset, net, scaler_hybrid, gridtype="coarse"):

        super(RelResids, self).__init__(testset, net, scaler_hybrid, gridtype)

        # compute the 2D array of all residuals
        self.rel_resid = (self.cont_true - self.cont_pred) / self.cont_true

        # compute summary statistics
        self.sigma_min = np.percentile(self.rel_resid, 100. * norm.cdf(-1.), axis=0)
        self.sigma_plus = np.percentile(self.rel_resid, 100. * norm.cdf(1.), axis=0)

        self.perc_median = np.percentile(self.rel_resid, 50., axis=0)

        print ("Computed summary statistics of residuals.")


class ResidualPlots(RelResids):
    def __init__(self, testset, net, scaler_hybrid, gridtype="coarse"):

        super(ResidualPlots, self).__init__(testset, net, scaler_hybrid, gridtype)


    def plotPercentiles(self, wave_min=1020., wave_max=1970., figsize=(6,4), dpi=320):
        '''
        Plot the median and 68% interval of residuals at every wavelength in the grid.

        @param wave_min:
        @param wave_max:
        @param figsize:
        @param dpi:
        @return:
        '''

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.plot(self.wave_grid, self.perc_median, label="Median", color="black")
        ax.fill_between(self.wave_grid, self.sigma_min, self.sigma_plus, alpha=0.3, label=r"68\% interval",
                        color="tab:orange")

        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)
        ax.set_xlabel("Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"$(F_{true} - F_{pred}) / F_{true}$")
        ax.set_title("Residuals Relative to True Continuum")

        ax.set_xlim(wave_min, wave_max)

        return fig, ax