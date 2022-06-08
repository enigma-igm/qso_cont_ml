import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
#from qso_fitting.models.utils.QuasarScaler import QuasarScaler
from utils.QuasarScaler import QuasarScaler
from utils.MinMaxScaler import MinMaxScaler
from utils.MedianScaler import MedianScaler
from utils.errorfuncs import WavWeights
from IPython import embed

def create_learners(parameters, learning_rate=0.1):
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    criterion = torch.nn.MSELoss(reduction="sum")

    return optimizer, criterion


def train_scalers(wave_grid, X_train, y_train, floorval=0.05):

    # calculate means and standard deviations
    X_mean = np.mean(X_train, axis=0)
    y_mean = np.mean(y_train, axis=0)
    X_std = np.std(X_train, axis=0) + floorval * np.median(X_mean)
    y_std = np.std(y_train, axis=0) + floorval * np.median(y_mean)

    # create the scalers
    scaler_X = QuasarScaler(wave_grid, X_mean, X_std)
    scaler_y = QuasarScaler(wave_grid, y_mean, y_std)

    return scaler_X, scaler_y


class Trainer:
    def __init__(self, net, optimizer, criterion, batch_size=1000, num_epochs=400):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def _train_glob_scalers(self, trainset, floorval=0.05, globscalers="both",\
                            scalertype="QuasarScaler"):
        '''Train the QuasarScalers on the training spectra and training continua.'''

        wave_grid = trainset.wave_grid.squeeze()
        flux = trainset.flux
        cont = trainset.cont

        if scalertype=="QuasarScaler":

            flux_mean = torch.mean(flux, dim=0)
            flux_std = torch.std(flux, dim=0) + floorval * torch.median(flux_mean)
            cont_mean = torch.mean(cont, dim=0)
            cont_std = torch.std(cont, dim=0) + floorval * torch.median(cont_mean)

            scaler_flux = QuasarScaler(wave_grid, flux_mean, flux_std)
            scaler_cont = QuasarScaler(wave_grid, cont_mean, cont_std)

        elif scalertype=="MinMaxScaler":

            flux_min = torch.min(flux)
            flux_max = torch.max(flux)
            cont_min = torch.min(cont)
            cont_max = torch.max(cont)

            scaler_flux = MinMaxScaler(flux_min, flux_max)
            scaler_cont = MinMaxScaler(cont_min, cont_max)

        elif scalertype=="MedianScaler":

            if (globscalers=="cont") & (len(flux.shape) > 2) & (flux.shape[1] > 1):
                ivar_extend = torch.unsqueeze(flux[:,1,:], dim=1)
                cont_ivar = torch.cat((cont, ivar_extend), dim=1)
                cont_mean = torch.mean(cont_ivar, dim=0)

            else:
                cont_mean = torch.mean(cont, dim=0)

            flux_mean = torch.mean(flux, dim=0)

            scaler_flux = MedianScaler(flux_mean, floorval)
            scaler_cont = MedianScaler(cont_mean, floorval)

        else:
            raise ValueError("Invalid scalertype given. Use QuasarScaler or MinMaxScaler instead.")

        if globscalers=="both":
            self.glob_scaler_flux = scaler_flux
            self.glob_scaler_cont = scaler_cont

        elif globscalers=="flux":
            self.glob_scaler_flux = scaler_flux
            self.glob_scaler_cont = scaler_flux

        elif globscalers=="cont":
            self.glob_scaler_flux = scaler_cont
            self.glob_scaler_cont = scaler_cont


    def train(self, trainset, validset, \
              savefile="simple_AdamW_net.pth", use_QSOScalers=True,\
              globscalers="both", weight=False, weightpower=1, scalertype="QuasarScaler"):
        '''Train the model.'''

        # first train the QSO scalers if use_QSOScalers==True
        if use_QSOScalers:

            self._train_glob_scalers(trainset, globscalers=globscalers, scalertype=scalertype)

        else:
            self.glob_scaler_flux = None
            self.glob_scaler_cont = None

        # set up the arrays for storing and checking the loss
        running_loss = np.zeros(self.num_epochs)
        valid_loss = np.zeros(self.num_epochs)
        min_valid_loss = np.inf

        self.wave_grid = trainset.wavegrid.squeeze()

        # set up DataLoaders
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=len(validset), shuffle=True)

        if weight:
            Weights = WavWeights(trainset.wave_grid, power=weightpower)
            weights_mse = Weights.weights_in_MSE
            weights_mse = weights_mse.to(self.device)

        # train the model
        for epoch in range(self.num_epochs):
            for flux_train_raw, _, cont_train_raw in train_loader:
                flux_train_raw = flux_train_raw.to(self.device)
                cont_train_raw = cont_train_raw.to(self.device)

                if use_QSOScalers:
                    flux_train = self.glob_scaler_flux.forward(flux_train_raw)
                    cont_train = self.glob_scaler_cont.forward(cont_train_raw)

                else:
                    flux_train = flux_train_raw
                    cont_train = cont_train_raw

                # set gradients to zero
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(flux_train)

                if use_QSOScalers:
                    outputs_real = self.glob_scaler_cont.backward(outputs)
                    outputs_real_rel = (outputs_real / cont_train_raw)
                    cont_train_rel = (cont_train_raw / cont_train_raw)

                else:
                    outputs_real_rel = (outputs / cont_train)
                    cont_train_rel = (cont_train / cont_train)

                if weight:
                    loss = self.criterion(outputs_real_rel * weights_mse, cont_train_rel * weights_mse)

                else:
                    loss = self.criterion(outputs_real_rel, cont_train_rel)

                loss.backward()
                self.optimizer.step()

                running_loss[epoch] += loss.item()

            print("Epoch " + str(epoch + 1) + "/" + str(self.num_epochs) + " completed.")

            # now use the validation set
            for flux_valid_raw, _, cont_valid_raw in valid_loader:

                flux_valid_raw = flux_valid_raw.to(self.device)
                cont_valid_raw = cont_valid_raw.to(self.device)

                if use_QSOScalers:
                    flux_valid = self.glob_scaler_flux.forward(flux_valid_raw)
                    cont_valid = self.glob_scaler_cont.forward(cont_valid_raw)

                else:
                    flux_valid = flux_valid_raw
                    cont_valid = cont_valid_raw

                # forward the network
                validoutputs = self.net(flux_valid)

                if use_QSOScalers:
                    validoutputs_real = self.glob_scaler_cont.backward(validoutputs)
                    validoutputs_real_rel = (validoutputs_real / cont_valid_raw)
                    cont_valid_rel = (cont_valid_raw / cont_valid_raw)

                else:
                    validoutputs_real_rel = (validoutputs / cont_valid)
                    cont_valid_rel = (cont_valid / cont_valid)

                if weight:
                    validlossfunc = self.criterion(validoutputs_real_rel * weights_mse, cont_valid_rel * weights_mse)

                else:
                    validlossfunc = self.criterion(validoutputs_real_rel, cont_valid_rel)

                valid_loss[epoch] += validlossfunc.item()

            valid_loss[epoch] = valid_loss[epoch] / len(validset)
            print("Validation loss: {:12.5f}".format(valid_loss[epoch]))

            # save the model if the validation loss decreases
            if min_valid_loss > valid_loss[epoch]:
                print("Validation loss decreased.")
                min_valid_loss = valid_loss[epoch]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "valid_loss": valid_loss[epoch],
                    "scaler_flux": self.glob_scaler_flux,
                    "scaler_cont": self.glob_scaler_cont
                }, savefile)


        # compute the loss per quasar
        running_loss = running_loss / len(trainset)

        # after completing the training route, load the model with lowest validation loss
        checkpoint = torch.load(savefile)
        self.net.load_state_dict(checkpoint["model_state_dict"])  # this should update net
        print("Best epoch:", checkpoint["epoch"])

        # save the diagnostics in the Trainer object
        self.training_loss = running_loss
        self.valid_loss = valid_loss


    def plot_loss(self, epoch_min=50, yscale="linear", titleadd=""):
        '''Plot the loss function for the training set and the validation set as a function of epoch number.'''

        epoch_no = np.arange(1, self.num_epochs+1)
        fig, ax = plt.subplots(figsize=(7,5), dpi=320)
        ax.plot(epoch_no, self.training_loss, alpha=0.7, label="Training")
        ax.plot(epoch_no, self.valid_loss, alpha=0.7, label="Validation")
        ax.set_xlim(xmin=epoch_min)
        if yscale=="linear":
            ymin = 0
        elif yscale=="log":
            ymin = 0.8
        else:
            print ("yscale must be 'linear' or 'log'.")
            return fig, ax
        max_loss_2show = self.valid_loss[epoch_min-1:].max()
        if max_loss_2show > 100*self.valid_loss.min():
            yscale = "log"
            ymin = 0.8
            print ("Large loss increase detected; yscale set to 'log'.")
        ax.set_ylim(ymin=ymin, ymax=max_loss_2show)
        ax.set_yscale(yscale)
        print ("ymax = "+str(self.valid_loss[epoch_min-1:].max()))
        ax.set_xlabel("Epoch number")
        ax.set_ylabel("Loss per quasar")
        ax.grid()

        fig.suptitle("MSE loss"+titleadd)
        ax.set_title("Summed over wavelengths, averaged over QSOs")
        ax.legend()

        return fig, ax


    def plot_sqrt_loss(self, epoch_min=50, yscale="linear", titleadd=""):
        '''Makes a plot of the square root of the MSE loss divided by the
        number of wavelength pixels, allowing for direct comparison with
        the mean relative residuals.'''

        n_lam = len(self.wave_grid)
        training_loss_avgd = self.training_loss / n_lam
        valid_loss_avgd = self.valid_loss / n_lam

        epoch_no = np.arange(1, self.num_epochs + 1)
        fig, ax = plt.subplots(figsize=(7, 5), dpi=320)
        ax.plot(epoch_no, np.sqrt(training_loss_avgd), alpha=0.7, label="Training")
        ax.plot(epoch_no, np.sqrt(valid_loss_avgd), alpha=0.7, label="Validation")
        ax.set_xlim(xmin=epoch_min)
        if yscale == "linear":
            ymin = 0
        elif yscale == "log":
            ymin = 0.8
        else:
            print("yscale must be 'linear' or 'log'.")
            return fig, ax

        max_loss_2show = 1.5*np.sqrt(valid_loss_avgd[epoch_min - 1:]).max()
        if max_loss_2show > 100 * self.valid_loss.min():
            yscale = "log"
            ymin = 0.8
            print("Large loss increase detected; yscale set to 'log'.")
        ax.set_ylim(ymin=ymin, ymax=max_loss_2show)
        ax.set_yscale(yscale)

        ax.set_xlabel("Epoch number")
        ax.set_ylabel(r"$\sqrt{MSE/n_\lambda}$")
        ax.grid()

        fig.suptitle("Square root of MSE loss" + titleadd)
        ax.set_title("Averaged over quasars and wavelength pixels")
        ax.legend()

        return fig, ax