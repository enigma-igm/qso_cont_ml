from models.linear_unet import LinearUNet
from data.load_datasets import SynthSpectra
from learning.learning_unet import UNetTrainer
from learning.learning import create_learners
from learning.testing import ResidualPlots, CorrelationMatrix, ModelResultsSpectra, RelResids
from data.load_data import load_synth_spectra, split_data, normalise_spectra
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.network import normalise
from data.load_data import load_synth_noisy_cont
from qso_fitting.data.sdss.paris import read_paris_continua
from scipy import interpolate
from data.load_data import load_paris_spectra
from utils.smooth_scaler import SmoothScaler
from pypeit.utils import fast_running_median

plt.rcParams["font.family"] = "serif"

# load the data
spectra = SynthSpectra(regridded=True, small=False, npca=10, noise=True, norm1280=True,\
                       forest=False, SN=10)
trainset, validset, testset = spectra.split()
wave_grid = spectra.wave_grid

# derive dimensions
n_feature = trainset.flux.shape[1]

# set whether we want to smooth the input in the last skip connection
smooth = True
no_final_skip = False
operator = "addition"

# set the hidden layer dimensions
layerdims = [100,200,300]

# initialize the simple network and train with the QSOScalers
unet = LinearUNet(n_feature, layerdims, activfunc="elu", operator=operator,\
                  no_final_skip=no_final_skip)
optimizer, criterion = create_learners(unet.parameters(), learning_rate=0.001)
trainer = UNetTrainer(unet, optimizer, criterion, num_epochs=300)
trainer.train(trainset, validset, use_QSOScalers=True, smooth=smooth, globscalers="cont",\
              weight=True, weightpower=1, loss_space="real-rel")

plotpath = "/net/vdesk/data2/buiten/MRP2/misc-figures/LinearUNet/"
plotpathadd = "/runmed-smoothing/"
filenamestart = plotpath+plotpathadd+"noforest_linweighted_contQSOScaler"
filenameend = "_28_01.png"

# plot the loss from the training routine
fig, ax = trainer.plot_sqrt_loss(epoch_min=1)
fig.show()
fig.savefig(filenamestart+"loss"+filenameend)

# run some tests on the test set
# use the hand-fit continua test set (half of it)
#wave_hf, cont_hf, flux_hf, _, _ = read_paris_continua()

# interpolate onto the hybrid grid
#cont_hf_hybrid = interpolate.interp1d(wave_hf, cont_hf, kind="cubic", bounds_error=False,\
#                                      fill_value="extrapolate", axis=1)(wave_grid)
#flux_hf_hybrid = interpolate.interp1d(wave_hf, flux_hf, kind="cubic", bounds_error=False,\
#                                      fill_value="extrapolate", axis=1)(wave_grid)

# use the hand-fit continua + simulated forest + simulated noise
#wave_hf, cont_hf, flux_hf = load_paris_spectra(noise=True)

# normalise the hand-fit spectra
#flux_test, cont_test = normalise_spectra(wave_hf, flux_hf, cont_hf)
#flux_test, cont_test = normalise_spectr#should not be necessary

# use the QSOScalers on input and output
#flux_test_scaled = trainer.scaler_X.forward(torch.FloatTensor(flux_test))
#cont_test_scaled = trainer.scaler_y.forward(torch.FloatTensor(cont_test))



# plot the residuals vs wavelength
resids = RelResids(testset, unet, scaler_flux=trainer.glob_scaler_flux,\
                   scaler_cont=trainer.glob_scaler_cont, smooth=smooth)
stats = ResidualPlots(testset, unet, scaler_flux=trainer.glob_scaler_flux,\
                           scaler_cont=trainer.glob_scaler_cont, smooth=smooth)
fig1, ax1 = stats.plot_means(show_std=False)
fig1.show()
#fig1.savefig(filenamestart+"residspec"+filenameend)

# plot the residuals in a histogram
fig2, ax2 = stats.resid_hist()
fig2.show()
#fig2.savefig(filenamestart+"residhist"+filenameend)

# plot the correlation matrix
corrmat = CorrelationMatrix(testset, unet, trainer.glob_scaler_flux,\
                            trainer.glob_scaler_cont, smooth=smooth)
fig3, ax3 = corrmat.show()
#fig3.savefig(filenamestart+"corrmat"+filenameend)


# plot a random result
testres = ModelResultsSpectra(testset, unet, scaler_flux=trainer.glob_scaler_flux,\
                       scaler_cont=trainer.glob_scaler_cont, smooth=smooth)
rand_indx = testres.random_index(4)
testres.create_figure(figsize=(15,10))
for i in range(len(rand_indx)):
    loc = int("22"+str(i+1))
    ax = testres.plot(rand_indx[i], subplotloc=loc)
testres.fig.suptitle("Test on synthetic spectra (npca=10)")

testres.show_figure()

testres2 = ModelResultsSpectra(testset, unet, scaler_flux=trainer.glob_scaler_flux,\
                        scaler_cont=trainer.glob_scaler_cont, smooth=smooth)
idx = testres2.random_index(1)
testres2.create_figure(figsize=(7,5))
testres2.plot(idx)
testres.fig.suptitle("Example of a predicted quasar continuum", size=15)

#testres.fig.savefig(filenamestart+"examples"+filenameend)
#rand_indx = np.random.randint(len(flux_test))
#rand_result_output = unet(flux_test_scaled[rand_indx])
#rand_result_descaled = trainer.scaler_y.backward(rand_result_output)
#rand_result = rand_result_descaled.detach().numpy()

#fig4, ax4 = plt.subplots(figsize=(7,5), dpi=320)
#ax4.plot(wave_grid, flux_test[rand_indx], alpha=0.8, lw=1, label="Input")
#ax4.plot(wave_grid, cont_test[rand_indx], alpha=0.7, lw=2, label="Target")
#ax4.plot(wave_grid, rand_result, alpha=0.8, lw=1, ls="--", label="Output")
#ax4.set_xlabel("Rest-frame wavelength ($\AA$)")
#ax4.set_ylabel("Normalised flux")
#ax4.legend()
#ax4.grid()
#ax4.set_title("Random example of a predicted QSO spectrum")
#fig4.show()