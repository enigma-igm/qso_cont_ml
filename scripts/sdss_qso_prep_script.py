'''Modeled after qso_fitting/data/sdss/create_sdss_autofit.py'''

import os
import numpy as np
from qso_fitting.data.sdss.sdss import sdss_qso_prep
import astropy.constants as const

lam_min = 980.
lam_max = 2040.

z_min = None
z_max = 4.
SN_min = 10.

dloglam = 1e-4
c_light = const.c.to("km/s").value
dv = dloglam * c_light * np.log(10)

outpath = os.getenv("SPECDB") + "/autofit/"
n_NN = None
NN = False

test_frac = 0.05

qsofile = "sdss_autofit_lam_min_{:d}_lam_max_{:d}.fits".format(int(np.round(lam_min)), int(np.round(lam_max)))
outfile = os.path.join(outpath, qsofile)

sdss_qso_prep(lam_min, lam_max, dv, SN_min, outfile, z_min=z_min, z_max=z_max, debug=True, NN=NN, n_NN=n_NN,
              test_fraction=test_frac)
