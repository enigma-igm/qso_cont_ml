import numpy as np
import matplotlib.pyplot as plt
import torch
import astropy.constants as const
from qso_fitting.data.sdss.sdss import sdss_data
from dw_inference.simulator.utils import get_blu_red_wave_grid
from data.load_data import normalise_spectra
from data.wavegrid_conversion import InputSpectra

import os
os.environ["SPECDB"] = "/net/vdesk/data2/buiten/MRP2/Data/"
os.environ["dw_inference"] = "/net/vdesk/data2/buiten/MRP2/code/dw_inference/dw_inference/"

# load the real spectra
wave_min = 1020.
wave_max = 1970.
SN_min = 5.
dloglam = 1.0e-4
c_light = (const.c.to("km/s")).value
dvpix = dloglam * c_light * np.log(10)
z_net = 2.8
z_min = z_net - 0.01
z_max = z_net + 0.01

zq, sn, meta, wave, flux, ivar, gpm = sdss_data(wave_min, wave_max, SN_min, dvpix,
                                                z_min, z_max, wave_norm=1280.)

# create the hybrid grid
#dvpix_red = 500.0
#wave_grid, dvpix_diff, ipix_blu, ipix_red = get_blu_red_wave_grid(1000., wave_max,
#                                                                  1216., dvpix,
#                                                                  dvpix_red)

inputspec = InputSpectra(wave, flux, ivar, zq, restframe=False,  wave_min=1000.,
                         wave_max=1970.)
inputspec.add_noise_channel()

fig, ax = plt.subplots(dpi=240)
ax.plot(inputspec.wave_grid, inputspec.flux[0,0], lw=.5)
ax.plot(inputspec.wave_grid, 1 / np.sqrt(inputspec.flux[0,1]), lw=.5,
        c="tab:green")
fig.show()

print ("Number of pixels with a bad ivar:", torch.sum(inputspec.flux[:,1] <= 0))