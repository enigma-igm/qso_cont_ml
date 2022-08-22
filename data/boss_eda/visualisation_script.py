from load import loadRedshiftLuminosityFile
from visualisation import RedshiftHistogram, LuminosityHistogram, RedshiftLuminosityHexbin
import matplotlib.pyplot as plt
import numpy as np

savepath = "/net/vdesk/data2/buiten/MRP2/misc-figures/BOSS_DR14_EDA/"

# load the data
redshifts, logLv = loadRedshiftLuminosityFile()

# create a figure for the hexbin/scatter plot and figures for the histograms
fig_hb = plt.figure(dpi=320)
fig_zhist = plt.figure(dpi=320)
fig_lumhist = plt.figure(dpi=320)

# create the objects
hexbin = RedshiftLuminosityHexbin(redshifts, logLv)
zhist = RedshiftHistogram(redshifts, logLv)
lumhist = LuminosityHistogram(logLv, redshifts)

# make the plots

fig_zhist, ax_zhist = zhist.plotInFigure(fig_zhist)
fig_lumhist, ax_lumhist = lumhist.plotInFigure(fig_lumhist)

ax_hb = fig_hb.add_subplot(111)
hb, cbar = hexbin.plotHexbin(ax_hb, fig_hb)
hexbin.plotScatter(ax_hb)

fig_hb.show()
fig_hb.savefig(savepath + "redshift-lum-hexbin.png")
plt.close()

fig_zhist.show()
fig_zhist.savefig(savepath + "redshift-hist.png")
plt.close()

fig_lumhist.show()
fig_lumhist.savefig(savepath + "lum-hist.png")
plt.close()