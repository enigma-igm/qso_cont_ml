from data.load_autofitter_datasets import AutofitterSpectra
from learning.testing_autofitters import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

path = "/net/vdesk/data2/buiten/MRP2/misc-figures/autofitter-tests/"

spectra = AutofitterSpectra(2.8)
resids_auto = AutofitterResidualPlots(spectra)
pred_spectra_auto = AutofitterPredictedSpectra(spectra)
corrmat_auto = AutofitterCorrelationMatrix(spectra)

resids_qsmooth = AutofitterResidualPlots(spectra, model="Qsmooth")

rnd_idx = pred_spectra_auto.random_index(1)

fig1, ax1 = resids_auto.plot_means(wave_lims=(1000., 2000.))
fig1.suptitle("Autofitter Performance")
fig1.show()
fig1.savefig(path+"autofitter-resids.png")

pred_spectra_auto.plot(rnd_idx, wave_lims=(1000., 2000.))
pred_spectra_auto.show_figure()

corrmat_auto.show(wave_lims=(1000., 2000.))

fig2, ax2 = resids_qsmooth.plot_means(wave_lims=(1000., 2000.))
fig2.suptitle("Qsmooth Performance")
fig2.show()
fig2.savefig(path+"qsmooth-resids.png")