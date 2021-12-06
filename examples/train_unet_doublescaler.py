from models.linear_unet import LinearUNet
from learning.learning_unet import DoubleScalingTrainer
from learning.learning import create_learners
from learning.testing_doublyscaled import ModelResultsDoublyScaled, ResidualStatisticsDoublyScaled
from data.load_datasets import SynthSpectra
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

# load the synthetic spectra with npca=10 and normalise to 1 around 1280 \AA
# use the SynthSpectra framework
synthspec = SynthSpectra(noise=True)
wave_grid = synthspec.wave_grid
trainset, validset, testset = synthspec.split()

# derive dimensions
n_feature = trainset.flux.shape[1]

# set the hidden layer dimensions
layerdims = [100,200,300]

# initialise the LinearUNet and train with the DoubleScalingTrainer
unet = LinearUNet(n_feature, layerdims, activfunc="elu", operator="addition",\
                  no_final_skip=True)
optimizer, criterion = create_learners(unet.parameters(), learning_rate=0.001)
trainer = DoubleScalingTrainer(unet, optimizer, criterion, num_epochs=300)
trainer.train_unet(trainset, validset)

# plot the loss from the training routine
fig, ax = trainer.plot_loss(epoch_min=1)
fig.show()

# plot the residuals
stats = ResidualStatisticsDoublyScaled(testset, unet, trainer.glob_scaler_flux,\
                                       trainer.glob_scaler_cont)
stats.compute_stats()
fig1, ax1 = stats.plot_means(show_std=False)
fig1.show()

# show some examples
testres = ModelResultsDoublyScaled(testset, unet, trainer.glob_scaler_flux,\
                                   trainer.glob_scaler_cont)
rand_indx = testres.random_index(4)
testres.create_figure(figsize=(15,10))
for i in range(len(rand_indx)):
    loc = int("22"+str(i+1))
    ax = testres.plot(rand_indx[i], subplotloc=loc)
testres.fig.suptitle("Test on synthetic spectra (npca=10)")

testres.show_figure()