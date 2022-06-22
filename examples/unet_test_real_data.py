import numpy as np
import matplotlib.pyplot as plt
import torch
import astropy.constants as const
from qso_fitting.data.sdss.sdss import sdss_data
from data.wavegrid_conversion import InputSpectra
from models.conv_unet import UNet
from learning.testing import ModelResultsSpectra

plt.rcParams["font.family"] = "serif"
plt.rcParams["text.usetex"] = True

import os
os.environ["SPECDB"] = "/net/vdesk/data2/buiten/MRP2/Data/"
os.environ["dw_inference"] = "/net/vdesk/data2/buiten/MRP2/code/dw_inference/dw_inference/"

# load the real spectra
wave_min = 1020.
wave_max = 1970.
SN_min = 1.
dloglam = 1.0e-4
c_light = (const.c.to("km/s")).value
dvpix = dloglam * c_light * np.log(10)
z_net = 2.8
z_min = z_net - 0.01
z_max = z_net + 0.01

zq, sn, meta, wave, flux, ivar, gpm = sdss_data(wave_min, wave_max, SN_min, dvpix,
                                                z_min, z_max, wave_norm=1280.)

# normalisation at 1280 A and rebinning onto the hybrid grid is done inside this class
inputspec = InputSpectra(wave, flux, ivar, zq, restframe=False,  wave_min=1000.,
                         wave_max=1970.)
inputspec.add_noise_channel()

'''
fig, ax = plt.subplots(dpi=240)
ax.plot(inputspec.wave_grid, inputspec.flux[0,0], lw=.5)
ax.plot(inputspec.wave_grid, 1 / np.sqrt(inputspec.flux[0,1]), lw=.5,
        c="tab:green")
fig.show()
'''

print ("Number of pixels with a bad ivar:", torch.sum(inputspec.flux[:,1] <= 0))

# now load the network we want to apply
basepath = "/net/vdesk/data2/buiten/MRP2/"
netpath = basepath + "/models/"
netfile = netpath + "2000epochs-forest-BOSSnoise-bothscalers-concatUNet-medianscaler-BOSS-layers4_8_16_32_64-maxpool2" \
                    "-ksize5_5_5_3_2-elu1-finalskip-enc_interp-refl_pad-lr1e-3-floorval1e-3.pth"

enc_chs = [2,4,8,16,32,64]
dec_chs = enc_chs[-1:0:-1]
conv_kernels_enc = [5,5,5,3,2]

net = UNet(len(inputspec.wave_grid), enc_chs, dec_chs, conv_kernels_enc, conv_kernels_enc[::-1],
           kernel_size_upconv=2, retain_dim=True, pool="max", pool_kernel_size=2,
           activfunc="elu", final_skip=True, padding_mode="reflect", crop_enc=False)
scaler_flux, scaler_cont = net.load(netfile)
scaler_flux.updateDevice()
scaler_cont.updateDevice()

# predict the continua and plot it for a random spectrum
modelspec = ModelResultsSpectra(inputspec, net, scaler_flux, scaler_cont)
rand_idx = modelspec.random_index(2)
modelspec.create_figure(figsize=(7,10))
axes = []
for i in range(len(rand_idx)):
    loc = int("21{}".format(i+1))
    ax = (modelspec.plot(rand_idx[i], subplotloc=loc, includesmooth=False))
    ax.set_title(r"Prediction for a spectrum with SN = {}".format(np.around(sn[rand_idx[i]], 2)))
    axes.append(ax)
modelspec.fig.suptitle(r"U-Net Continuum Prediction on Real Spectra of $2.79 < z < 2.81$", size=15)

figpath = "/net/vdesk/data2/buiten/MRP2/misc-figures/thesis-figures/results/"

modelspec.fig.savefig("{}real-spec_predictions_SNmin{}_22_06.png".format(figpath, SN_min), bbox_inches="tight")
modelspec.fig.savefig("{}real-spec_predictions_SNmin{}_22_06.pdf".format(figpath, SN_min), bbox_inches="tight")
modelspec.fig.show()