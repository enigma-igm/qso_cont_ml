'''Module for making the plots of redshift and logLv in the BOSS DR14 data.'''

import numpy as np
from data.boss_eda.load import loadRedshiftLuminosityFile
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def RiceRule(n_samples):
    '''
    Applies the Rice Rule to determine the number of histogram bins to use.

    @param n_samples: int
    @return:
        n_bins: int
    '''

    n_bins_float = 2 * n_samples ** (1./3)
    n_bins = int(np.ceil(n_bins_float))

    return n_bins


class HistogramBase:
    '''
    Base class for constructing a histogram of any 1D set of data points.

    Attributes:
        data:
        n_samples:
        n_bins:
        counts:
        edges:
        widths:
        mids:

    Methods:
        plotOnAxis
    '''

    def __init__(self, data, range=None):
        '''
        @param data: ndarray of shape (n_samples,)
        @param range: (float, float) or NoneType
        '''

        assert isinstance(data, np.ndarray)
        assert data.ndim == 1
        self.data = data
        self.n_samples = data.size
        self.n_bins = RiceRule(self.n_samples)

        self.counts, self.edges = np.histogram(self.data, bins=self.n_bins, range=range)

        self.widths = self.edges[1:] - self.edges[:-1]
        self.mids = self.edges[:-1] + 0.5 * self.widths


    def plotOnAxis(self, ax, alpha=.7, label=""):
        '''
        Plot the histogram on a provided axis.

        @param ax: matplotlib Axes instance
        @param alpha: float
        @param label: float
        @return:
            ax: matplotlib Axes instance
        '''

        ax.bar(self.mids, self.counts, self.widths, alpha=alpha, label=label)

        # might want to put the layout somewhere else?
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)

        ax.set_ylabel("Occurrences")
        ax.legend()

        return ax


class RedshiftHistogram(HistogramBase):
    '''
    Class for constructing a histogram of redshifts and adding it to a figure.

    Attributes:
        data:
        n_samples:
        n_bins:
        counts:
        edges:
        widths:
        mids:
        logLv_min:
        logLv_max:
        label:

    Methods:
        plotOnAxis:
        plotInFigure:
    '''

    def __init__(self, redshifts, logLv, logLv_lims=None, range=None):
        '''
        @param redshifts:
        @param logLv:
        @param logLv_lims:
        @param range:
        '''

        assert redshifts.shape == logLv.shape

        if logLv_lims is None:
            redshifts_use = redshifts
            self.logLv_min = logLv.min()
            self.logLv_max = logLv.max()

        #elif (isinstance(logLv_lims, tuple) or isinstance(logLv_lims, list) or isinstance(logLv_lims, np.ndarray)) & \
        #    len(logLv_lims) == 2:
        elif len(logLv_lims) == 2:
            self.logLv_min = logLv_lims[0]
            self.logLv_max = logLv_lims[1]

            sel = (logLv > self.logLv_min) & (logLv < self.logLv_max)
            redshifts_use = redshifts[sel]

        else:
            raise TypeError("logLv_lims must indicate the minimum and maximum value of logLv to select.")

        self.label = r"{} < $\log L_\nu$ < {}".format(np.around(self.logLv_min, 2), np.around(self.logLv_max, 2))

        super(RedshiftHistogram, self).__init__(redshifts_use, range)


    def createFigure(self, figsize=(6,4), dpi=320):
        '''
        Create a figure. Wrapper for plt.figure.

        @param figsize:
        @param dpi:
        @return:
        '''

        fig = plt.figure(figsize=figsize, dpi=dpi)

        return fig


    def plotInFigure(self, fig, subplotloc=111):
        '''
        Add this object's histogram to a figure in a subplot.

        @param fig:
        @param subplotloc:
        @return:
        '''

        ax = fig.add_subplot(subplotloc)
        ax = self.plotOnAxis(ax, label=self.label)
        ax.set_xlabel("Redshifts")

        return fig, ax


    def quickPlot(self, figsize=(6,4), dpi=320):
        '''
        Convenience method for making a simple plot containing a single histogram.

        @param figsize:
        @param dpi:
        @return:
        '''

        fig = self.createFigure(figsize, dpi)
        fig, ax = self.plotInFigure(fig)

        return fig, ax