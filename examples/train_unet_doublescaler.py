from models.linear_unet import LinearUNet
from learning.learning_unet import DoubleScalingTrainer
from learning.learning import create_learners
from learning.testing_doublyscaled import ModelResultsDoublyScaled, ResidualStatisticsDoublyScaled
from data.load_datasets import SynthSpectra
import matplotlib.pyplot as plt
from learning.testing_doublyscaled import DoubleScalingResidStats, DoubleScalingCorrelationMatrix, DoubleScalingResultsSpectra
from models.network import Net

plt.rcParams["font.family"] = "serif"

# load the synthetic spectra with npca=10 and normalise to 1 around 1280 \AA
# use the SynthSpectra framework
synthspec = SynthSpectra(noise=True, forest=False, window=20, newnorm=False,\
                         homosced=True, poisson=False, SN=10)
wave_grid = synthspec.wave_grid
trainset, validset, testset = synthspec.split()

# test on npca = 15
#synthspec15 = SynthSpectra(noise=True, npca=15)
#_, _, testset = synthspec15.split()

# derive dimensions
n_feature = trainset.flux.shape[1]

# set the hidden layer dimensions
layerdims = [300, 200, 100]

# initialise the LinearUNet and train with the DoubleScalingTrainer
#unet = Net(n_feature, 3*n_feature, n_feature)
unet = LinearUNet(n_feature, layerdims, activfunc="elu", operator="addition",\
                  no_final_skip=True)
optimizer, criterion = create_learners(unet.parameters(), learning_rate=0.01)
trainer = DoubleScalingTrainer(unet, optimizer, criterion, num_epochs=150)
trainer.train_unet(trainset, validset, loss_space="real-rel",\
                   globscalers="cont", relscaler=True, weight=True,\
                   weightpower=1, relglobscaler=True,\
                   abs_descaling=False)

savefolder = "/net/vdesk/data2/buiten/MRP2/misc-figures/LinearUNet/double-scaling/cont-better-noise/"
filenamestart = savefolder + "homoscedSN10_regsmooth_linweighted_contQSOScaler_"
filenameend = "_28_01.png"

# plot the loss from the training routine
# plot the square root of the loss per wavelength pixel
fig, ax = trainer.plot_sqrt_loss(epoch_min=1)
fig.show()
fig.savefig(filenamestart+"loss"+filenameend)

# plot the residuals
#stats = ResidualStatisticsDoublyScaled(testset, unet, trainer.glob_scaler_flux,\
#                                       trainer.glob_scaler_cont)
#stats.compute_stats()
stats = DoubleScalingResidStats(testset, unet, trainer.glob_scaler_flux,\
                                trainer.glob_scaler_cont)
fig0, ax0 = stats.resid_hist()
fig0.show()
fig0.savefig(filenamestart+"residhist"+filenameend)
fig1, ax1 = stats.plot_means(show_std=False)
fig1.show()
fig1.savefig(filenamestart+"residspec"+filenameend)

# plot the correlation matrix
corrmat = DoubleScalingCorrelationMatrix(testset, unet, trainer.glob_scaler_flux,\
                                         trainer.glob_scaler_cont)
fig2, ax2 = corrmat.show()
fig2.savefig(filenamestart+"corrmat"+filenameend)

# show some examples
#testres = ModelResultsDoublyScaled(testset, unet, trainer.glob_scaler_flux,\
#                                   trainer.glob_scaler_cont)
testres = DoubleScalingResultsSpectra(testset, unet, trainer.glob_scaler_flux,\
                                      trainer.glob_scaler_cont)
rand_indx = testres.random_index(4)
testres.create_figure(figsize=(12,8))
for i in range(len(rand_indx)):
    loc = int("22"+str(i+1))
    ax = testres.plot(rand_indx[i], subplotloc=loc, includesmooth=True,\
                      plotinput=True, plottarget=True)
testres.fig.suptitle("Test on synthetic spectra (npca=10)")

testres.show_figure()
testres.fig.savefig(filenamestart+"examples"+filenameend)

# also plot the raw network output for the same test quasars
testres_raw = DoubleScalingResultsSpectra(testset, unet, trainer.glob_scaler_flux,\
                                      trainer.glob_scaler_cont)
testres_raw.create_figure(figsize=(12,9))
for i in range(len(rand_indx)):
    loc = int("22"+str(i+1))
    ax = testres_raw.plot_doublyscaled(rand_indx[i], subplotloc=loc)
testres_raw.fig.suptitle("Network output in doubly scaled space")

#testres_raw.show_figure()
testres_raw.fig.savefig(filenamestart+"examplesraw"+filenameend)

# plot the network output in locally scaled space
testres_loc = DoubleScalingResultsSpectra(testset, unet, trainer.glob_scaler_flux,\
                                          trainer.glob_scaler_cont)
testres_loc.create_figure(figsize=(12,9))
for i in range(len(rand_indx)):
    loc = int("22"+str(i+1))
    ax = testres_loc.plot_smoothscaled(rand_indx[i], subplotloc=loc)
testres_loc.fig.suptitle("Network output in locally scaled space")
testres_loc.fig.savefig(filenamestart+"examplesloc"+filenameend)

testpreds_compare = DoubleScalingResultsSpectra(testset, unet, trainer.glob_scaler_flux,\
                                                trainer.glob_scaler_cont)
testpreds_compare.plot_raw_preds(100)
testpreds_compare.fig.suptitle("Random predictions on the test set (with forest & homoscedastic noise)")
testpreds_compare.fig.savefig(filenamestart+"examplesraw-comparison"+filenameend)

testpreds_compare2 = DoubleScalingResultsSpectra(testset, unet, trainer.glob_scaler_flux,\
                                                 trainer.glob_scaler_cont)
testpreds_compare2.raw_preds_means()
testpreds_compare2.fig.suptitle("Test set predictions with forest and homoscedastic noise")
testpreds_compare2.fig.savefig(filenamestart+"examplesraw-distr"+filenameend)
