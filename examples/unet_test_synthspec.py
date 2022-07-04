import numpy as np
import matplotlib.pyplot as plt
import torch
import astropy.constants as const
from qso_fitting.data.sdss.sdss import sdss_data
from data.wavegrid_conversion import InputSpectra
from data.load_datasets import SynthSpectra
from models.conv_unet import UNet
from learning.testing import ModelResultsSpectra
from learning.testing import ResidualPlots

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

import os
os.environ["SPECDB"] = "/net/vdesk/data2/buiten/MRP2/Data/"
os.environ["dw_inference"] = "/net/vdesk/data2/buiten/MRP2/code/dw_inference/dw_inference/"

npca = 10

spec = SynthSpectra(test=True, npca=10)
spec.add_noise_channel()

# now load the network we want to apply
basepath = "/net/vdesk/data2/buiten/MRP2/"
netpath = basepath + "/models/"
netfile = netpath + "2000epochs-forest-BOSSnoise-bothscalers-concatUNet-medianscaler-BOSS-layers4_8_16_32_64-maxpool2" \
                    "-ksize5_5_5_3_2-elu1-finalskip-enc_interp-refl_pad-lr1e-3-floorval1e-3.pth"

enc_chs = [2,4,8,16,32,64]
dec_chs = enc_chs[-1:0:-1]
conv_kernels_enc = [5,5,5,3,2]

net = UNet(len(spec.wave_grid), enc_chs, dec_chs, conv_kernels_enc, conv_kernels_enc[::-1],
           kernel_size_upconv=2, retain_dim=True, pool="max", pool_kernel_size=2,
           activfunc="elu", final_skip=True, padding_mode="reflect", crop_enc=False)
scaler_flux, scaler_cont = net.load(netfile)
scaler_flux.updateDevice()
scaler_cont.updateDevice()

'''
# predict the continua and plot it for a random spectrum
modelspec = ModelResultsSpectra(spec, net, scaler_flux, scaler_cont)
rand_idx = modelspec.random_index(2)
modelspec.create_figure(figsize=(7,10))
axes = []
for i in range(len(rand_idx)):
    loc = int("21{}".format(i+1))
    ax = (modelspec.plot(rand_idx[i], subplotloc=loc, includesmooth=False))
    ax.set_title(r"Prediction for a spectrum constructed from 10 PCA vectors")
    axes.append(ax)
modelspec.fig.suptitle(r"U-Net Continuum Prediction on Synthetic Spectra of $z = 2.8$", size=15)
'''

figpath = "/net/vdesk/data2/buiten/MRP2/misc-figures/thesis-figures/results/"

'''
#modelspec.fig.savefig("{}synth-spec_predictions_npca{}.png".format(figpath, npca), bbox_inches="tight")
#modelspec.fig.savefig("{}synth-spec_predictions_npca{}.pdf".format(figpath, npca), bbox_inches="tight")
modelspec.fig.show()

# plot the residuals
resids = ResidualPlots(spec, net, scaler_flux, scaler_cont)
fig, ax = resids.plot_percentiles()
ax.set_ylim(-.2, .2)
fig.suptitle("Network Performance on Synthetic Test Spectra ({} PCA vectors)".format(npca), size=15)

#fig.savefig("{}synth-spec_residuals_npca{}.png".format(figpath, npca), bbox_inches="tight")
#fig.savefig("{}synth-spec_residuals_npca{}.pdf".format(figpath, npca), bbox_inches="tight")
fig.show()
'''

# also plot the residuals of output interpolated onto a uniform grid
resids_uni = ResidualPlots(spec, net, scaler_flux, scaler_cont, interpolate=True)
fig2, ax2 = resids_uni.plot_percentiles(figsize=(6,3.5))
ax2.set_ylim(-.9, .9)
fig2.suptitle("Network Performance on Synthetic Test Spectra", size=15)
ax2.set_title("After interpolating output onto uniform grid")
fig2.savefig("{}synth-spec_residuals_npca{}_unigrid_alt.pdf".format(figpath, npca), bbox_inches="tight")
fig2.show()