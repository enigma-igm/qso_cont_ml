
import os
import numpy as np
from matplotlib import pyplot as plt
from qso_fitting.utils.get_paths import get_sdss_autofit_path
from qso_fitting.data.utils import read_sdss_autofits
from qso_fitting.data.utils import inverse
from astropy.io import fits

lam_min = 980.0
lam_max = 2040.0
outpath = get_sdss_autofit_path()# os.path.join(outpath, 'sdss_training_data/', 'sdss_autofit.fits')
qsofile = 'sdss_autofit_lam_min_{:d}_lam_max_{:d}.fits'.format(int(np.round(lam_min)), int(np.round(lam_max)))
outfile = os.path.join(outpath,qsofile)
wave, cont, cont_ivar, flux, ivar, gpm, meta = read_sdss_autofits(outfile, include_raw=True)



