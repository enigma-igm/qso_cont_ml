import torch
import numpy as np
from utils.MedianScaler import MedianScaler
from utils.errorfuncs import WavWeights
from torch.utils.data import DataLoader
from data.load_data_new import SynthSpectra


class UNetTrainer:
    def __init__(self, net, optimizer, criterion, batch_size=1000, num_epochs=500):

        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def trainScalers(self, trainset, floorval=0.05):
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

        # the coarse-grid scaler only needs the coarse-grid continuum
        cont_coarse_unsq = torch.unsqueeze(trainset.cont_coarse, dim=1)
        mean_cont_coarse = torch.mean(cont_coarse_unsq, dim=0)

        # initialise the two scalers
        scaler_hybrid = MedianScaler(mean_spec_hybrid, floorval)
        scaler_coarse = MedianScaler(mean_cont_coarse, floorval)

        return scaler_hybrid, scaler_coarse


    def train(self, trainset, validset, savefile, scaler_floorval=0.05):
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
        self.scaler_hybrid, self.scaler_coarse = self.trainScalers(trainset, scaler_floorval)

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

                # scale the input and target output
                flux_input_train = self.scaler_hybrid.forward(flux_input_raw)
                true_cont_train = self.scaler_coarse.forward(true_cont_raw)

                # set gradients to zero
                self.optimizer.zero_grad()

                # forward the network
                outputs = self.net(flux_input_train)

                # compute the weighted loss
                outputs_real_rel = self.scaler_coarse.backward(outputs) / true_cont_raw
                targets_rel = true_cont_raw / true_cont_raw
                loss = self.criterion(outputs_real_rel * weights_mse, targets_rel * weights_mse)

                loss.backward()

                # optimize
                self.optimizer.step()

                # store the training loss
                running_loss[epoch] += loss.item()

            print("Epoch {}/{} completed.".format(epoch+1, self.num_epochs))

            # now compute the validation loss
            for valid_input_raw, valid_cont_raw in valid_loader:

                valid_input_raw = valid_input_raw.to(self.device)
                valid_cont_raw = valid_cont_raw.to(self.device)

                valid_input_scaled = self.scaler_hybrid.forward(valid_input_raw)

                valid_outputs = self.net(valid_input_scaled)
                valid_outputs_real_rel = self.scaler_coarse.backward(valid_outputs) / valid_cont_raw
                valid_targets_rel = valid_cont_raw / valid_cont_raw

                validlossfunc = self.criterion(valid_outputs_real_rel * weights_mse, valid_targets_rel * weights_mse)
                valid_loss[epoch] += validlossfunc.item()

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
                    "scaler_coarse": self.scaler_coarse,
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


# TO DO: write loss plotting method or class for plotting loss
# potentially allow for restarting of the training routine?