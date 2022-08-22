from load import loadRedshiftLuminosityFile
from visualisation import binEdges, LuminosityHistogram, RedshiftHistogram
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
savepath = "/net/vdesk/data2/buiten/MRP2/misc-figures/BOSS_DR14_EDA/"
SN_min = 10

redshifts, logLv = loadRedshiftLuminosityFile()

z_bin_edges = binEdges(redshifts, 0.08)
print ("Redshift bin edges: {}".format(z_bin_edges))
print ("Number of redshift bins: {}".format(z_bin_edges.size - 1))

# first plot the marginal distribution of redshifts with the selected bin edges
z_hist = RedshiftHistogram(redshifts, logLv, logLv_lims=np.percentile(logLv, [0.1, 99.9]), bins=z_bin_edges)

fig, ax = z_hist.quickPlot()

fig.suptitle("Distribution of Redshifts for BOSS DR14 QSOs", size=15)
ax.set_title(r"SN > {}; $\Delta z = 0.08$".format(SN_min))

fig.savefig("{}redshift-hist-dz0.08.png".format(savepath))
plt.close()

# now plot the conditional distribution of logLv for each bin
for i in range(z_bin_edges.size - 1):
    lum_hist = LuminosityHistogram(logLv, redshifts, redshift_lims=z_bin_edges[i:i+2],
                                   range=np.percentile(logLv, [0.1, 99.9]))
    fig, ax = lum_hist.quickPlot()
    ax.set_xlim(np.percentile(logLv, [0.1, 99.9]))
    fig.suptitle("Conditional Distribution of Lyman-Limit Luminosities", size=15)
    ax.set_title(lum_hist.label)
    fig.savefig("{}{}z_bin{}.png".format(savepath, "conditional_hists/", i+1))
    plt.close()
    #plt.show(block=True)
    #plt.close()