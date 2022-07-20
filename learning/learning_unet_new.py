import torch
import numpy as np
from utils.MedianScaler import MedianScaler
from utils.errorfuncs import WavWeights
from torch.utils.data import DataLoader
from data.load_data_new import SynthSpectra
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class UNetTrainer:
    '''
    Trainer for the convolutional UNet. Assumes that input contains absorption spectra, ivar noise vectors and
    mean transmitted flux vectors. Also assumes the input is on the hybrid grid while the output is on a coarse grid.
    The scaling is done by MedianScalers.
    '''

    def __init__(self, net, optimizer, criterion, batch_size=1000, num_epochs=500):

        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def trainScalers(self, trainset, floorval=0.05, scale_trans=True):
        '''
        Train the MedianScalers based on the true continua. We train both a hybrid-grid scaler and a coarse-grid scaler.

        @param trainset:
        @param floorval:
        @return:
        '''

        if not isinstance(trainset, SynthSpectra):
            raise TypeError("Parameter 'trainset' must be an instance of SynthSpectra.")

        # load the hybrid grid continuum and attach the ivar and mean transmission
        cont_hybrid_unsq = torch.unsqueeze(trainset.cont_hybrid, dim=1)
        ivar_hybrid_unsq = torch.unsqueeze(trainset.ivar_hybrid, dim=1)
        mean_trans_unsq = torch.unsqueeze(trainset.mean_trans_hybrid, dim=1)

        # compute the mean spectra
        input_hybrid_scaler = torch.cat([cont_hybrid_unsq, ivar_hybrid_unsq, mean_trans_unsq], dim=1)
        mean_spec_hybrid = torch.mean(input_hybrid_scaler, dim=0)

        '''TO DO: remove the coarse-grid scaler'''

        '''
        # the coarse-grid scaler only needs the coarse-grid continuum
        cont_coarse_unsq = torch.unsqueeze(trainset.cont_coarse, dim=1)
        mean_cont_coarse = torch.mean(cont_coarse_unsq, dim=0)
        '''

        # initialise the two scalers
        scaler_hybrid = MedianScaler(mean_spec_hybrid, floorval, scale_trans=scale_trans)
        #scaler_coarse = MedianScaler(mean_cont_coarse, floorval)

        return scaler_hybrid


    def train(self, trainset, validset, savefile, scaler_floorval=0.05, scale_trans=True):
        '''
        Train the U-Net.

        @param trainset:
        @param validset:
        @param savefile:
        @param scaler_floorval:
        @return:
        '''

        # TO DO: write error messages
        assert isinstance(trainset, SynthSpectra)
        assert isinstance(validset, SynthSpectra)

        # train the hybrid-grid and coarse-grid scalers
        self.scaler_hybrid = self.trainScalers(trainset, scaler_floorval, scale_trans=scale_trans)

        # check the dimensions of the scalers
        '''
        print ("Shape of scaler_hybrid.mean_spectrum:", self.scaler_hybrid.mean_spectrum.shape)
        print ("Shape of scaler_coarse.mean_spectrum:", self.scaler_coarse.mean_spectrum.shape)
        print ("Median in scaler_hybrid:", self.scaler_hybrid.median)
        '''

        # set up tensors for storing the loss
        running_loss = torch.zeros(self.num_epochs)
        valid_loss = torch.zeros(self.num_epochs)
        min_valid_loss = np.inf

        # set up DataLoaders
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=len(validset), shuffle=True)

        # set up velocity width weights
        Weights = WavWeights(trainset.wave_hybrid)
        weights_mse = Weights.weights_in_MSE.to(self.device)

        self.wave_hybrid = trainset.wave_hybrid
        self.wave_coarse = trainset.wave_coarse

        # train the model
        for epoch in range(self.num_epochs):
            for flux_input_raw, true_cont_raw in train_loader:

                # transfer everything to the set device
                flux_input_raw = flux_input_raw.to(self.device)
                true_cont_raw = true_cont_raw.to(self.device)

                #print ("Number of NaN input values before scaling:", torch.sum(torch.isnan(flux_input_raw)))
                #print ("Shape of input before scaling:", flux_input_raw.shape)

                # scale the input and target output
                flux_input_train = self.scaler_hybrid.forward(flux_input_raw)

                #print ("Number of NaN input values:", torch.sum(torch.isnan(flux_input_train)))

                # set gradients to zero
                self.optimizer.zero_grad()

                # forward the network
                outputs = self.net(flux_input_train)

                #print ("Number of NaN output values:", torch.sum(torch.isnan(outputs)))

                # compute the weighted loss
                outputs_real_rel = self.scaler_hybrid.backward(outputs) / true_cont_raw

                #print ("Number of NaN output values after descaling:", torch.sum(torch.isnan(outputs_real_rel)))
                targets_rel = true_cont_raw / true_cont_raw
                loss = self.criterion(outputs_real_rel * weights_mse, targets_rel * weights_mse)

                loss.backward()

                # optimize
                self.optimizer.step()

                # store the training loss
                running_loss[epoch] += loss.item()

                #print ("Running training loss:", running_loss[epoch])

            print("Epoch {}/{} completed.".format(epoch+1, self.num_epochs))

            # now compute the validation loss
            for valid_input_raw, valid_cont_raw in valid_loader:

                valid_input_raw = valid_input_raw.to(self.device)
                valid_cont_raw = valid_cont_raw.to(self.device)

                valid_input_scaled = self.scaler_hybrid.forward(valid_input_raw)

                valid_outputs = self.net(valid_input_scaled)
                valid_outputs_real_rel = self.scaler_hybrid.backward(valid_outputs) / valid_cont_raw
                valid_targets_rel = valid_cont_raw / valid_cont_raw

                validlossfunc = self.criterion(valid_outputs_real_rel * weights_mse, valid_targets_rel * weights_mse)
                valid_loss[epoch] += validlossfunc.item()

                #print ("Valid loss:", valid_loss[epoch])

            # normalise the validation loss to the number of quasars in the validation set
            valid_loss[epoch] = valid_loss[epoch] / len(validset)
            print ("Validation loss: {:12.5f}".format(valid_loss[epoch]))

            # save the model if the validation loss decreases
            if min_valid_loss > valid_loss[epoch]:
                print ("Validation loss decreased.")
                min_valid_loss = valid_loss[epoch]

                torch.save({
                    "epoch": epoch+1,
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "valid_loss": valid_loss[epoch],
                    "scaler_hybrid": self.scaler_hybrid,
                }, savefile)

        # compute the loss per quasar once training is complete
        running_loss = running_loss / len(trainset)

        # load the model with the lowest validation loss
        checkpoint = torch.load(savefile)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        print ("Best epoch:", checkpoint["epoch"])

        # save the diagnostics as attributes
        self.training_loss = running_loss
        self.valid_loss = valid_loss


    def plotSqRtLoss(self, epoch_min=50, yscale="linear", titleadd=""):
        '''
        Plot the square root of the MSE loss averaged over both quasars and wavelength pixels, allowing for a direct
        comparison with the relative residuals on the test set. The loss is plotted for both the training set and the
        validation set.

        @param epoch_min:
        @param yscale:
        @param titleadd:
        @return:
        '''

        n_lam = len(self.wave_coarse)
        training_loss_avgd = self.training_loss / n_lam
        valid_loss_avgd = self.valid_loss / n_lam

        epoch_no = np.arange(1, self.num_epochs + 1)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=320)

        ax.plot(epoch_no, training_loss_avgd, alpha=.7, label="Training")
        ax.plot(epoch_no, valid_loss_avgd, alpha=.7, label="Validation")
        ax.set_xlim(xmin=epoch_min)

        max_loss = self.valid_loss[epoch_min - 1:].max()
        if max_loss > 100 * self.valid_loss.min():
            print("Large loss increase detected; yscale set to 'log'.")

        if yscale == "linear":
            ymin = 0
        elif yscale == "log":
            ymin = 0.8
        else:
            raise ValueError("yscale must be 'linear' or 'log'.")

        ax.set_ylim(ymin=ymin, ymax=max_loss)
        ax.set_yscale(yscale)
        print("ymax = {}".format(self.valid_loss[epoch_min - 1:].max()))

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlabel("Epoch number")
        ax.set_ylabel(r"$\sqrt{MSE / n_\lambda}$")
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)

        fig.suptitle("Square Root of MSE Loss{}".format(titleadd))
        ax.set_title("Averaged over QSOs and pixels")
        ax.legend()

        return fig, ax


    def plotLoss(self, epoch_min=50, yscale="linear", titleadd=""):
        '''
        Plot the MSE loss for the training set and the validation set as a function of epoch.
        This method plots the loss summed over wavelength pixels and averaged over quasars.
        May become deprecated.

        @param epoch_min:
        @param yscale:
        @param titleadd:
        @return:
        '''

        epoch_no = np.arange(1, self.num_epochs+1)

        fig, ax = plt.subplots(figsize=(6,4), dpi=320)

        ax.plot(epoch_no, self.training_loss, alpha=.7, label="Training")
        ax.plot(epoch_no, self.valid_loss, alpha=.7, label="Validation")
        ax.set_xlim(xmin=epoch_min)

        max_loss = self.valid_loss[epoch_min-1:].max()
        if max_loss > 100 * self.valid_loss.min():
            print ("Large loss increase detected; yscale set to 'log'.")

        if yscale == "linear":
            ymin = 0
        elif yscale == "log":
            ymin = 0.8
        else:
            raise ValueError("yscale must be 'linear' or 'log'.")

        ax.set_ylim(ymin=ymin, ymax=max_loss)
        ax.set_yscale(yscale)
        print ("ymax = {}".format(self.valid_loss[epoch_min-1:].max()))

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xlabel("Epoch number")
        ax.set_ylabel("Loss per quasar")
        ax.grid(which="major", alpha=.3)
        ax.grid(which="minor", alpha=.1)

        fig.suptitle("MSE loss{}".format(titleadd))
        ax.set_title("Summed over wavelengths, averaged over QSOs")
        ax.legend()

        return fig, ax

# potentially allow for restarting of the training routine?