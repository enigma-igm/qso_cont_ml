from models.network import Net
from learning.learning import create_learners, train_model, test_model
from learning.testing import CorrelationMatrix, ResidualStatistics
from data.load_data import load_synth_spectra, split_data
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
#from utils.errorfuncs import corr_matrix_relresids

plt.rcParams["font.family"] = "serif"

# first train on the npca=6 set
# then test on the npca=15 set
# also perform the splitting for better comparison
wave_grid, qso_cont_npca6, qso_flux_npca6 = load_synth_spectra(small=False, npca=6)
X_train6, X_valid6, X_test6, y_train6, y_valid6, y_test6 = split_data(qso_flux_npca6, qso_cont_npca6)
wave_grid15, qso_cont_npca15, qso_flux_npca15 = load_synth_spectra(small=False, npca=15)
X_train15, X_valid15, X_test15, y_train15, y_valid15, y_test15 = split_data(qso_flux_npca15, qso_cont_npca15)

n_feature = len(X_train6[1])
n_output = len(y_train6[1])

net = Net(n_feature, 100, n_output)
optimizer, criterion = create_learners(net.parameters())
running_loss, mse_loss_valid, scaler_X, scaler_y = train_model(wave_grid, X_train6, y_train6,\
                                                               X_valid6, y_valid6, net, optimizer,\
                                                               criterion, batch_size=1000, num_epochs=400)
epochs = np.arange(1, len(running_loss)+1)

# plot the test statistics as a function of wavelength
Stats = ResidualStatistics(X_test6, y_test6, scaler_X, scaler_y, net)
fig0, ax0 = Stats.plot_means(wave_grid)
fig0.show()

# test the final model and print the result
#mse_test, corr_matrix = test_model(X_test, y_test, scaler_X, scaler_y, net)
#print ("MSE on test set:", mse_test)

fig, ax = plt.subplots(figsize=(7,5), dpi=320)
ax.plot(epochs, running_loss, label="Training set")
ax.plot(epochs, mse_loss_valid, label="Validation set")
ax.legend()
ax.set_xlabel("Epoch number")
ax.set_ylabel("MSE")
ax.set_yscale("log")
ax.set_title("Mean squared error on the normalised spectra")
fig.show()

# now plot an example result on the npca = 15 TEST set
rand_indx = np.random.randint(len(X_test15))
rescaled_result = net.full_predict(X_test15[rand_indx], scaler_X, scaler_y)
#test_input_normed = normalise(scaler_X, X_test[rand_indx])
#test_input_normed_var = Variable(torch.FloatTensor(test_input_normed.numpy()))
#normed_result = net(test_input_normed_var)
#rescaled_result = scaler_y.backward(normed_result)
fig2, ax2 = plt.subplots(figsize=(7,5), dpi=320)
ax2.plot(wave_grid, X_test15[rand_indx], alpha=0.8, lw=2, label="Input")
ax2.plot(wave_grid, y_test15[rand_indx], alpha=0.8, lw=2, label="Target")
ax2.plot(wave_grid, rescaled_result, alpha=0.8, lw=2, label="Output")
ax2.set_xlabel("Rest-frame wavelength ($\AA$)")
ax2.set_ylabel("Flux (a.u.)")
ax2.legend()
ax2.grid()
ax2.set_title("Example of a predicted quasar spectrum")
fig2.show()

# visualise the correlation matrix for the npca = 15 TEST set
CorrMat = CorrelationMatrix(X_test15, y_test15, scaler_X, scaler_y, net)
CorrMat.show(wave_grid)
#fig3, ax3 = plt.subplots()
#im = ax3.pcolormesh(wave_grid, wave_grid, corr_matrix)
#fig3.show()