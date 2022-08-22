'''Module for making the plots of redshift and logLv in the BOSS DR14 data.'''

import numpy as np
#from data.boss_eda.load import loadRedshiftLuminosityFile
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


def uniformBinEdges(data, width):
    '''
    Determine edges of bins for histograms or binned data in the case of uniform bin width.

    @param data: ndarray of shape (n_samples,)
    @param width: float or int
    @return:
        edges: ndarray of shape (n_bins + 1,)
    '''

    data_min = data.min()
    data_max = data.max()

    n_bins = int(np.ceil((data_max - data_min) / width))

    edges = np.array([data_min + i * width for i in range(n_bins)])

    return edges


def binEdges(data, widths):
    '''
    Determine edges of bins for histograms or binned data.

    @param data: ndarray of shape (n_samples,)
    @param widths: ndarray of shape (n_bins,) or float or int
    @return:
        edges: ndarray of shape (n_bins + 1,)
    '''

    if isinstance(widths, np.ndarray) or isinstance(widths, tuple) or isinstance(widths, list):
        if len(widths) > 1:
            edges = data.min() + widths
        elif len(widths) == 1:
            edges = uniformBinEdges(data, widths[0])
        else:
            raise TypeError("Widths must be either a single positive number or an array of edges.")

    elif isinstance(widths, int) or isinstance(widths, float):
        edges = uniformBinEdges(data, widths)

    return edges


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

    def __init__(self, data, range=None, bins=None):
        '''
        @param data: ndarray of shape (n_samples,)
        @param range: (float, float) or NoneType
        @param bins: NoneType or int or ndarray of shape (n_samples + 1) or str
            If NoneType, the Rice Rule is used to determine the number of bins. Otherwise, 'bins' is passed directly
            onto numpy.histogram.
        '''

        assert isinstance(data, np.ndarray)
        assert data.ndim == 1
        self.data = data
        self.n_samples = data.size

        # if bins is None, use the Rice Rule to determine the number of bins
        if bins is None:
            self.n_bins = RiceRule(self.n_samples)
            self.counts, self.edges = np.histogram(self.data, bins=self.n_bins, range=range)

        # otherwise pass bins directly
        else:
            self.counts, self.edges = np.histogram(self.data, bins=bins, range=range)
            self.n_bins = self.counts.size

        self.widths = self.edges[1:] - self.edges[:-1]
        self.mids = self.edges[:-1] + 0.5 * self.widths


    def plotOnAxis(self, ax, alpha=.7, label="", orientation="vertical"):
        '''
        Plot the histogram on a provided axis.

        @param ax: matplotlib Axes instance
        @param alpha: float
        @param label: float
        @return:
            ax: matplotlib Axes instance
        '''

        ax.bar(self.mids, self.counts, self.widths, alpha=alpha, label=label, orientation=orientation)

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

    def __init__(self, redshifts, logLv, logLv_lims=None, range=None, bins=None):
        '''
        @param redshifts:
        @param logLv:
        @param logLv_lims:
        @param range:
        @param bins:
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

        super(RedshiftHistogram, self).__init__(redshifts_use, range, bins)


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
        ax.set_xlabel("Redshift")

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


class LuminosityHistogram(HistogramBase):
    '''
    Class for plotting histograms of logLv.
    '''

    def __init__(self, logLv, redshifts, redshift_lims=None, range=None, bins=None):

        assert redshifts.shape == logLv.shape

        if redshift_lims is None:
            logLv_use = logLv
            self.redshift_min = redshifts.min()
            self.redshift_max = redshifts.max()

        elif len(redshift_lims) == 2:
            self.redshift_min = redshift_lims[0]
            self.redshift_max = redshift_lims[1]

            sel = (redshifts > self.redshift_min) & (redshifts < self.redshift_max)
            logLv_use = logLv[sel]

        else:
            raise TypeError("redshift_lims must indicate the minimum and maximum redshift to select.")

        self.label = r"{} < $z$ < {}".format(np.around(self.redshift_min, 2), np.around(self.redshift_max, 2))

        super(LuminosityHistogram, self).__init__(logLv_use, range, bins)


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
        ax.set_xlabel(r"$\log L_\nu$")

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


class RedshiftLuminosityHexbin:

    def __init__(self, redshifts, logLv, redshift_lims=None, logLv_lims=None):

        if redshift_lims is None:
            z_sel = np.ones_like(redshifts, dtype=bool)
            self.redshift_min = redshifts.min()
            self.redshift_max = redshifts.max()

        else:
            self.redshift_min, self.redshift_max = redshift_lims
            z_sel = (redshifts > self.redshift_min) & (redshifts < self.redshift_max)

        if logLv_lims is None:
            logLv_sel = np.ones_like(logLv, dtype=bool)
            self.logLv_min = logLv.min()
            self.logLv_max = logLv.max()

        else:
            self.logLv_min, self.logLv_max = logLv_lims
            logLv_sel = (logLv > self.logLv_min) & (logLv < self.logLv_max)

        self.redshifts = redshifts[z_sel & logLv_sel]
        self.logLv = logLv[z_sel & logLv_sel]


    def createFigure(self, figsize=(6,4), dpi=320):

        fig = plt.figure(figsize=figsize, dpi=dpi)

        return fig


    def plotHexbin(self, ax, fig, gridsize=50):

        if gridsize is None:
            gridsize = 2 * int(np.ceil(np.sqrt(RiceRule(self.redshifts.size))))

        hb = ax.hexbin(self.redshifts, self.logLv, cmap="turbo", gridsize=gridsize, edgecolors="none")

        ax.set_xlabel("Redshift")
        ax.set_ylabel(r"$\log L_\nu$")

        ax.set_xlim(self.redshifts.min(), self.redshifts.max())
        ax.set_ylim(self.logLv.min(), self.logLv.max())

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        cbar = fig.colorbar(hb, ax=ax, label="Occurrences")

        return hb, cbar


    def plotScatter(self, ax):

        ax.plot(self.redshifts, self.logLv, ls="", marker="o", color="black", alpha=.2, markersize=1)