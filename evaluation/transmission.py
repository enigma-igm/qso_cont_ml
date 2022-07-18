'''Module for computing and plotting the mean transmission from predicted continua.'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from data.load_data_new import SynthSpectra
from evaluation.testing_new import ModelResults
from evaluation.utils import bootstrapMean

class MeanTransmission(ModelResults):
    '''
    Class for computing the mean transmission given the noiseless absorption spectrum and the network's predicted
    continua.

    Currently only supports synthetic test spectra (for which we know the noiseless absorption spectrum).
    Only the fine grid is included.

    TODO:
        Add redshift selection
        Add error margins

    Attributes:
        device: torch.device
        input_hybrid: torch tensor of shape (n_qso, 3, n_wav)
        flux: ndarray of shape (n_qso, n_wav)
        noise: ndarray of shape (n_qso, n_wav)
        cont_pred: ndarray of shape (n_qso, n_wav)
        cont_true: ndarray of shape (n_qso, n_wav)
        wave_grid: ndarray of shape (n_wav,)
        grid_type: str
        mean_trans_pred: ndarray of shape (n_wav,)
        mean_trans_true: ndarray of shape (n_wav,)
    '''

    def __init__(self, testset, net, scaler_hybrid):

        super(MeanTransmission, self).__init__(testset, net, scaler_hybrid, gridtype="fine")

        # load the noiseless absorption spectra and compute the transmission for each QSO
        flux_noiseless = testset.noiseless_flux_fine.cpu().detach().numpy()
        trans_pred = flux_noiseless / self.cont_pred
        trans_true = flux_noiseless / self.cont_true

        # compute the mean over all spectra
        self.mean_trans_pred = np.mean(trans_pred, axis=0)
        self.mean_trans_true = np.mean(trans_true, axis=0)

        #TODO: add non-parametric bootstrap algorithm for error margins?
        self.sigma_min_pred, self.sigma_plus_pred = bootstrapMean(trans_pred)
        self.sigma_min_true, self.sigma_plus_true = bootstrapMean(trans_true)



class MeanTransmissionPlot(MeanTransmission):
    '''
    Class for plotting the mean transmission, given the network predictions.
    '''

    def __init__(self, testset, net, scaler_hybrid):

        super(MeanTransmissionPlot, self).__init__(testset, net, scaler_hybrid)

        # initialise a figure placeholder variable and empty axes
        self.fig = None
        self.axes = []


    def createFigure(self, figsize=(6,4), dpi=320):
        '''
        Create a figure to add subplots to.

        @param figsize:
        @param dpi:
        @return:
        '''

        self.fig = plt.figure(figsize=figsize, dpi=dpi)

        return self.fig


    def plot(self, figsize=(6,4), dpi=320, subplotloc=111, alpha=.7, contpredcolor="darkred", wave_min=1020.,
             wave_max=1970.):
        '''

        @param figsize:
        @param dpi:
        @param subplotloc:
        @param alpha:
        @param wave_min:
        @param wave_max:
        @return:
        '''

        if self.fig is None:
            fig = self.createFigure(figsize=figsize, dpi=dpi)
        else:
            fig = self.fig

        ax = fig.add_subplot(subplotloc)

        ax.plot(self.wave_grid, self.mean_trans_true, alpha=alpha, lw=1.5, c="tab:orange", label="Ground truth")
        ax.fill_between(self.wave_grid, self.sigma_min_true, self.sigma_plus_true, alpha=.3, c="tab:orange")

        ax.plot(self.wave_grid, self.mean_trans_pred, alpha=alpha, lw=1., c=contpredcolor, label="Prediction")
        ax.fill_between(self.wave_grid, self.sigma_min_pred, self.sigma_plus_pred, alpha=.3, c=contpredcolor)

        ax.axvline(1216., alpha=0.7, lw=1., ls="dashdot", color="black", label="Blue-red split")

        ax.set_xlabel(r"Rest-frame wavelength ($\AA$)")
        ax.set_ylabel(r"$F_{abs} / F_{cont}$")
        ax.legend()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)

        ax.set_xlim(wave_min, wave_max)
        ax.set_ylim(-.2, 1.2)

        ax.set_title("Input truth vs. network prediction")

        self.axes.append(ax)

        return ax


    def showFigure(self):
        '''
        Show the figure.

        @return:
        '''

        self.fig.suptitle("Mean Transmission", size=15)
        self.fig.show()