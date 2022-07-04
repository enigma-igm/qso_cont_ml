from data.load_autofitter_datasets import AutofitterSpectra
from learning.testing_autofitters import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

path = "/net/vdesk/data2/buiten/MRP2/misc-figures/autofitter-tests/29_06_"

spectra = AutofitterSpectra(2.8, test=True)

resids_auto = AutofitterResidualPlots(spectra)
pred_spectra_auto = AutofitterPredictedSpectra(spectra)
#corrmat_auto = AutofitterCorrelationMatrix(spectra)


resids_qsmooth = AutofitterResidualPlots(spectra, model="Qsmooth")


rnd_idx = pred_spectra_auto.random_index(1)

#fig1, ax1 = resids_auto.plot_means(wave_lims=(1020., 1970.))
fig1, ax1 = resids_auto.plot_percentiles(wave_lims=(1020., 1970.), figsize=(6,3.5))
ax1.set_xlim(1020., 1970.)
ax1.set_ylim(-.9, .9)
fig1.suptitle("Autofitter Performance", size=15)
fig1.show()
fig1.savefig(path+"autofitter-resids.png")
fig1.savefig(path+"autofitter-resids_alt.pdf", bbox_inches="tight")

pred_spectra_auto.plot(rnd_idx, wave_lims=(1020., 1970.))
pred_spectra_auto.axes[0].set_ylim(ymin=0)
pred_spectra_auto.show_figure()

#corrmat_auto.show(wave_lims=(1000., 2000.))

fig2, ax2 = resids_qsmooth.plot_percentiles(wave_lims=(1020., 1970.), figsize=(6,3.5))
ax2.set_xlim(1020., 1970.)
ax2.set_ylim(-.9, .9)
#fig2, ax2 = resids_qsmooth.plot_means(wave_lims=(1000., 2000.))
fig2.suptitle("QSmooth Performance", size=15)
fig2.show()
fig2.savefig(path+"qsmooth-resids.png")
fig2.savefig(path+"qsmooth-resids_alt.pdf", bbox_inches="tight")