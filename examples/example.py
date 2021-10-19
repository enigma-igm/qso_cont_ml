import os
from models.network import Net, normalise
from learning.learning import create_learners, train_model, test_model, Trainer
from learning.testing import ResidualStatistics, CorrelationMatrix
from data.load_data import load_synth_spectra, split_data, normalise_spectra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
#from utils.errorfuncs import corr_matrix_relresids

plt.rcParams["font.family"] = "serif"

wave_grid, qso_cont, qso_flux = load_synth_spectra(small=False)
# normalise the spectra
qso_flux, qso_cont = normalise_spectra(wave_grid, qso_flux, qso_cont)
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(qso_flux, qso_cont, train_size=0.9, test_size=0.05)

n_feature = len(X_train[1])
n_output = len(y_train[1])

net = Net(n_feature, 100, n_output)
optimizer, criterion = create_learners(net.parameters())
trainer = Trainer(net, optimizer, criterion, batch_size=1000, num_epochs=400)
trainer.train(wave_grid, X_train, y_train, X_valid, y_valid)
#running_loss, mse_loss_valid, scaler_X, scaler_y = train_model(wave_grid, X_train, y_train,\
#                                                               X_valid, y_valid,net, optimizer,\
#                                                               criterion, batch_size=1000, num_epochs=500)
#epochs = np.arange(1, len(running_loss)+1)

# test the final model and print the result

#mse_test, corr_matrix = test_model(X_test, y_test, trainer.scaler_X, trainer.scaler_y, net)
#print ("MSE on test set:", mse_test)
corrmat = CorrelationMatrix(X_test, y_test, trainer.scaler_X, trainer.scaler_y, net)


# plot the loss from the training route
fig, ax = trainer.plot_loss(yscale="linear", epoch_min=100)
fig.show()

# compute and plot statistics
stats = ResidualStatistics(X_test, y_test, trainer.scaler_X, trainer.scaler_y, net)
fig0, ax0 = stats.plot_means(wave_grid, show_std=False)
fig0.show()

fig1, ax1 = stats.resid_hist()
fig1.show()

#fig, ax = plt.subplots(figsize=(7,5), dpi=320)
#ax.plot(epochs, running_loss, label="Training set")
#ax.plot(epochs, mse_loss_valid, label="Validation set")
#ax.legend()
#ax.set_xlabel("Epoch number")
#ax.set_ylabel("MSE per quasar")
#ax.set_yscale("log")
#ax.set_title("Mean squared error on the normalised spectra")
#fig.show()

# now plot an example result
rand_indx = np.random.randint(len(X_test))
rescaled_result = net.full_predict(X_test[rand_indx], trainer.scaler_X, trainer.scaler_y)
#test_input_normed = normalise(scaler_X, X_test[rand_indx])
#test_input_normed_var = Variable(torch.FloatTensor(test_input_normed.numpy()))
#normed_result = net(test_input_normed_var)
#rescaled_result = scaler_y.backward(normed_result)
fig2, ax2 = plt.subplots(figsize=(7,5), dpi=320)
ax2.plot(wave_grid, X_test[rand_indx], alpha=0.8, lw=2, label="Input")
ax2.plot(wave_grid, y_test[rand_indx], alpha=0.8, lw=2, label="Target")
ax2.plot(wave_grid, rescaled_result, alpha=0.8, lw=2, ls="--", label="Output")
ax2.set_xlabel("Rest-frame wavelength ($\AA$)")
ax2.set_ylabel("Flux (a.u.)")
ax2.legend()
ax2.grid()
ax2.set_title("Example of a predicted quasar spectrum")
fig2.show()

# visualise the correlation matrix
fig3, ax3 = corrmat.show(wave_grid)
#fig3, ax3 = plt.subplots()
#im = ax3.pcolormesh(wave_grid, wave_grid, corr_matrix)
#ax3.set_xlabel("Rest-frame wavelength ($\AA$)")
#ax3.set_ylabel("Rest-frame wavelength ($\AA$)")
#ax3.set_title("Correlation matrix")
#cbar = fig3.colorbar(im, ax=ax3, label="Correlation")
#fig3.show()