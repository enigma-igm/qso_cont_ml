import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import torch

from data.load_datasets import SynthSpectra
from models.conv_unet import UNet
from learning.learning_unet import UNetTrainer
from learning.learning import create_learners
from learning.testing import ModelResultsSpectra, ResidualPlots

spectra = SynthSpectra(noise=True, forest=False)
spectra.add_channel_shape(1)
trainset, validset, testset = spectra.split()
wave_grid = spectra.wave_grid
print ("Shape of trainset flux:", trainset.flux.shape)

n_ftrs = trainset.flux.shape[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = UNet(n_ftrs, retain_dim=True, num_class=1, enc_chs=(1,64,128), dec_chs=(128,64))
optimizer, criterion = create_learners(net.parameters(), learning_rate=0.1)
trainer = UNetTrainer(net, optimizer, criterion, num_epochs=20, batch_size=2500)
trainer.train(trainset, validset, use_QSOScalers=True, smooth=False,\
              globscalers="cont", weight=False, loss_space="real-rel")

fig, ax = trainer.plot_sqrt_loss(epoch_min=1)
fig.show()

resids = ResidualPlots(testset, net, scaler_flux=trainer.glob_scaler_flux,\
                       scaler_cont=trainer.glob_scaler_cont)
fig1, ax1 = resids.plot_means(show_std=False)

testres = ModelResultsSpectra(testset, net, scaler_flux=trainer.glob_scaler_flux,\
                       scaler_cont=trainer.glob_scaler_cont, smooth=False)
rand_indx = testres.random_index(4)
testres.create_figure(figsize=(15,10))
for i in range(len(rand_indx)):
    loc = int("22"+str(i+1))
    ax = testres.plot(rand_indx[i], subplotloc=loc, includesmooth=False)
testres.fig.suptitle("Test on synthetic spectra (npca=10)")

testres.show_figure()