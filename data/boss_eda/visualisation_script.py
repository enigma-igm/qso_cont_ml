from load import loadRedshiftLuminosityFile
from visualisation import RedshiftHistogram, LuminosityHistogram, RedshiftLuminosityHexbin
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

savepath = "/net/vdesk/data2/buiten/MRP2/misc-figures/BOSS_DR14_EDA/filtered/"
SN_min = 10

# load the data
redshifts, logLv = loadRedshiftLuminosityFile()

# create a figure for the hexbin/scatter plot and figures for the histograms
fig_hb = plt.figure(dpi=320)
fig_zhist = plt.figure(dpi=320)
fig_lumhist = plt.figure(dpi=320)

logLv_lims = np.percentile(logLv, [.1,99.9])

# create the objects
hexbin = RedshiftLuminosityHexbin(redshifts, logLv, logLv_lims=None)
#hexbin = RedshiftLuminosityHexbin(redshifts, logLv)
zhist = RedshiftHistogram(redshifts, logLv, logLv_lims=None)
lumhist = LuminosityHistogram(logLv, redshifts, range=None)

# make the plots

fig_zhist, ax_zhist = zhist.plotInFigure(fig_zhist)
fig_lumhist, ax_lumhist = lumhist.plotInFigure(fig_lumhist)

ax_hb = fig_hb.add_subplot(111)
hb, cbar = hexbin.plotHexbin(ax_hb, fig_hb, gridsize=30)
#hexbin.plotScatter(ax_hb)

fig_zhist.suptitle("Distribution of Redshifts for BOSS DR14 QSOs", size=15)
ax_zhist.set_title("SN_min > {}".format(SN_min))

fig_lumhist.suptitle("Distribution of Lyman-Limit Luminosities", size=15)
ax_lumhist.set_title("SN_min > {}".format(SN_min))

fig_hb.suptitle("Redshifts vs. Lyman-Limit Luminosities", size=15)
ax_hb.set_title("SN_min > {}".format(SN_min))

fig_hb.show()
fig_hb.savefig(savepath + "redshift-lum-hexbin-full-new.png")
plt.close()

fig_zhist.show()
fig_zhist.savefig(savepath + "redshift-hist-full-new.png")
plt.close()

fig_lumhist.show()
fig_lumhist.savefig(savepath + "lum-hist-full-new.png")
plt.close()