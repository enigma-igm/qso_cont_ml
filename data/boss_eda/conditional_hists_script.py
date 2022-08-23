from load import loadRedshiftLuminosityFile
from visualisation import binEdges, LuminosityHistogram, RedshiftHistogram
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"
savepath = "/net/vdesk/data2/buiten/MRP2/misc-figures/BOSS_DR14_EDA/"
SN_min = 10

redshifts, logLv = loadRedshiftLuminosityFile()

dz = 0.1
z_bin_edges = binEdges(redshifts, dz)
print ("Redshift bin edges: {}".format(z_bin_edges))
print ("Number of redshift bins: {}".format(z_bin_edges.size - 1))

# first plot the marginal distribution of redshifts with the selected bin edges
z_hist = RedshiftHistogram(redshifts, logLv, logLv_lims=np.percentile(logLv, [0.1, 99.9]), bins=z_bin_edges)

fig, ax = z_hist.quickPlot()

fig.suptitle("Distribution of Redshifts for BOSS DR14 QSOs", size=15)
ax.set_title(r"SN > {}; $\Delta z = 0.08$".format(SN_min))

fig.savefig("{}redshift-hist-dz{}.png".format(savepath, dz))
plt.close()

# now plot the conditional distribution of logLv for each bin
# take dlogLv = 0.1

dlogLv = 0.2
logLv_edges = binEdges(logLv, dlogLv, data_range=np.percentile(logLv, [0.1, 99.9]))
print ("Number of logLv bins: {}".format(logLv_edges.size - 1))

for i in range(z_bin_edges.size - 1):

    #lum_hist = LuminosityHistogram(logLv, redshifts, redshift_lims=z_bin_edges[i:i+2],
    #                               range=np.percentile(logLv, [0.1, 99.9]))
    lum_hist = LuminosityHistogram(logLv, redshifts, redshift_lims=z_bin_edges[i:i+2],
                                   range=np.percentile(logLv, [0.1, 99.9]), bins=logLv_edges)
    fig, ax = lum_hist.quickPlot()
    ax.set_xlim(np.percentile(logLv, [0.1, 99.9]))
    fig.suptitle("Conditional Distribution of Lyman-Limit Luminosities", size=15)
    ax.set_title(r"$\Delta \log L_\nu = $ {}".format(dlogLv))
    fig.savefig("{}{}dlogL{}z_bin{}.png".format(savepath, "conditional_hists_dz0.1/", dlogLv, i+1))
    plt.close()
    #plt.show(block=True)
    #plt.close()