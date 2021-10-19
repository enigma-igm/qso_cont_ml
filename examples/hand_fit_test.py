# file for training on PCA-generated continua and testing on hand-fit continua

import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_synth_spectra, split_data, normalise_spectra
from models.network import Net
from learning.learning import Trainer, create_learners, train_scalers
from learning.testing import ResidualStatistics, CorrelationMatrix
import torch
plt.rcParams["font.family"] = "serif"

# first load and split the synthetic spectra with npca = 10
wave_grid, qso_cont, qso_flux = load_synth_spectra()
# normalise the synthetic spectra
qso_flux, qso_cont = normalise_spectra(wave_grid, qso_flux, qso_cont)
splitted = split_data(qso_flux, qso_cont)
X_train, X_valid, X_test, y_train, y_valid, y_test = splitted

# now load the hand-fit spectra
path = "/net/vdesk/data2/buiten/MRP2/code/qso_cont_ml/"
handfitfile = "/data/paris_noiselessflux_regridded.npy"
handfitdata = np.load(path+handfitfile)
wave_grid_handfit = handfitdata[0,:,0]
cont_handfit = handfitdata[:,:,1]
flux_handfit = handfitdata[:,:,2]

# normalise the hand-fit spectra
flux_handfit, cont_handfit = normalise_spectra(wave_grid_handfit, flux_handfit, cont_handfit)

# compare the two grids
print ("Training grid:", wave_grid.shape)
print ("Testing grid:", wave_grid_handfit.shape)

# train the model on the training set (with help of validation set)
n_features = len(wave_grid)
size_hidden = 100
model = Net(n_features, size_hidden, n_features)
optimizer, criterion = create_learners(model.parameters())
trainer = Trainer(model, optimizer, criterion)
trainer.train(wave_grid, X_train, y_train, X_valid, y_valid)

# plot the loss
fig, ax = trainer.plot_loss()
fig.show()

# load the last model
#modelfile = "/net/vdesk/data2/buiten/MRP2/code/qso_cont_ml/examples/saved_model.pth"
#model.load_state_dict(torch.load(modelfile)["model_state_dict"])

# hack for now: train separate scalers for the hand-fit spectra
#scaler_flux_handfit, scaler_cont_handfit = train_scalers(wave_grid_handfit, flux_handfit, cont_handfit)

# evaluate
residstats = ResidualStatistics(flux_handfit, cont_handfit, trainer.scaler_X,\
                                trainer.scaler_y, model)
fig1, ax1 = residstats.plot_means(wave_grid_handfit)
fig1.show()

fig2, ax2 = residstats.resid_hist()
fig2.show()

corrmat = CorrelationMatrix(flux_handfit, cont_handfit, trainer.scaler_X,\
                            trainer.scaler_y, model)
corrmat.show(wave_grid_handfit)

# plot a random example
rand_indx = np.random.randint(low=0, high=len(cont_handfit))
result = model.full_predict(flux_handfit[rand_indx], trainer.scaler_X,\
                           trainer.scaler_y)
fig3, ax3 = plt.subplots(figsize=(7,5), dpi=320)
ax3.plot(wave_grid_handfit, flux_handfit[rand_indx], alpha=0.8, label="Input", lw=1)
ax3.plot(wave_grid_handfit, cont_handfit[rand_indx], alpha=0.8, label="Target")
ax3.plot(wave_grid_handfit, result, alpha=0.8, label="Output", ls="--")
ax3.set_xlabel("Wavelength (Angstrom)")
ax3.set_ylabel("Flux")
ax3.legend()
ax3.grid()
fig3.show()