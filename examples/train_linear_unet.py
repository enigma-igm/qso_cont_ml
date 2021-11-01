from models.linear_unet import LinearUNet
from learning.learning_unet import UNetTrainer
from learning.learning import create_learners
from learning.testing import ResidualStatistics, CorrelationMatrix
from data.load_data import load_synth_spectra, split_data, normalise_spectra
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.network import normalise

plt.rcParams["font.family"] = "serif"

# load the synthetic spectra with npca=10 and normalise to 1 around 1280 \AA
wave_grid, qso_cont, qso_flux = load_synth_spectra(small=False)
flux_norm, cont_norm = normalise_spectra(wave_grid, qso_flux, qso_cont)

# split into training set, validation set and test set
flux_train, flux_valid, flux_test, cont_train, cont_valid, cont_test = split_data(flux_norm,\
                                                                                  cont_norm)
# derive dimensions
n_feature = flux_train.shape[1]

# initialize the simple network and train WITHOUT using the QSOScalers
unet = LinearUNet(n_feature, [100,200,300], activfunc="elu", operator="multiplication")
optimizer, criterion = create_learners(unet.parameters(), learning_rate=0.0001)
trainer = UNetTrainer(unet, optimizer, criterion)
trainer.train(wave_grid, flux_train, cont_train, flux_valid, cont_valid, use_QSOScalers=True)

# plot the loss from the training routine
fig, ax = trainer.plot_loss(epoch_min=0)
ax.set_ylim(300,600)
fig.show()

# run some tests on the test set
# use the QSOScalers on input and output
flux_test_scaled = trainer.scaler_X.forward(torch.FloatTensor(flux_test))
cont_test_scaled = trainer.scaler_y.forward(torch.FloatTensor(cont_test))

# plot the residuals vs wavelength
stats = ResidualStatistics(flux_test, cont_test, scaler_flux=trainer.scaler_X,\
                           scaler_cont=trainer.scaler_y, net=unet)
fig1, ax1 = stats.plot_means(wave_grid, show_std=False)
fig1.show()

# plot the residuals in a histogram
fig2, ax2 = stats.resid_hist()
fig2.show()

# plot the correlation matrix
corrmat = CorrelationMatrix(flux_test, cont_test, trainer.scaler_X, trainer.scaler_y, unet)
fig3, ax3 = corrmat.show(wave_grid)

# plot a random result
rand_indx = np.random.randint(len(flux_test))
rand_result_output = unet(flux_test_scaled[rand_indx])
rand_result_descaled = trainer.scaler_y.backward(rand_result_output)
rand_result = rand_result_descaled.detach().numpy()

fig4, ax4 = plt.subplots(figsize=(7,5), dpi=320)
ax4.plot(wave_grid, flux_test[rand_indx], alpha=0.8, lw=1, label="Input")
ax4.plot(wave_grid, cont_test[rand_indx], alpha=0.7, lw=2, label="Target")
ax4.plot(wave_grid, rand_result, alpha=0.8, lw=1, ls="--", label="Output")
ax4.set_xlabel("Rest-frame wavelength ($\AA$)")
ax4.set_ylabel("Normalised flux")
ax4.legend()
ax4.grid()
ax4.set_title("Random example of a predicted QSO spectrum")
fig4.show()