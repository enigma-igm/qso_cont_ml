# code for training the LinearUNet model
import torch
from torch.autograd import Variable
from sklearn.utils import shuffle
import numpy as np
from learning.learning import Trainer
from models.linear_unet import get_rel_resids
from models.network import normalise
from pypeit.utils import fast_running_median
from utils.smooth_scaler import SmoothScaler, DoubleScaler
from qso_fitting.models.utils import QuasarScaler
from data.load_datasets import SynthSpectra
from torch.utils.data import DataLoader

class UNetTrainer(Trainer):
    def __init__(self, net, optimizer, criterion, batch_size=1000, num_epochs=400):
        super(UNetTrainer, self).__init__(net, optimizer, criterion, batch_size=batch_size, num_epochs=num_epochs)


    def _train_DoubleScalers(self, wave_grid, flux_train, cont_train,\
                             smoothwindow=20, floorval=0.05):

        doubscaler_flux = DoubleScaler(wave_grid, flux_train, smoothwindow=smoothwindow,\
                                       floorval=floorval)
        doubscaler_cont = DoubleScaler(wave_grid, flux_train, smoothwindow=smoothwindow,\
                                       floorval=floorval, cont_train=cont_train)

        self.scaler_X = doubscaler_flux
        self.scaler_y = doubscaler_cont


    def train(self, wave_grid, X_train, y_train, X_valid, y_valid,\
              savefile="LinearUNet.pth", use_QSOScalers=False, smooth=False,\
              use_DoubleScalers=False):
        '''DoubleScaler training currently does not work properly!'''

        # do the smoothing before applying the QSOScalers
        if smooth:
            # smooth the input for the last skip connection
            X_train_smooth, X_valid_smooth = np.zeros(X_train.shape), np.zeros(X_valid.shape)
            for i in range(len(X_train)):
                X_train_smooth[i] = fast_running_median(X_train[i], 20)
            for i in range(len(X_valid)):
                X_valid_smooth[i] = fast_running_median(X_valid[i], 20)

        # use the QSOScaler
        if use_QSOScalers:
            self.train_QSOScalers(wave_grid, X_train, y_train)
            X_train = self.scaler_X.forward(torch.FloatTensor(X_train))
            y_train = self.scaler_y.forward(torch.FloatTensor(y_train))
            X_valid = self.scaler_X.forward(torch.FloatTensor(X_valid))
            y_valid = self.scaler_y.forward(torch.FloatTensor(y_valid))

            # also apply QSOScalers to the smoothed input
            # note: we may need to train new QSOScalers for smoothed spectra
            if smooth:
                X_train_smooth = self.scaler_X.forward(torch.FloatTensor(X_train_smooth))
                X_valid_smooth = self.scaler_X.forward(torch.FloatTensor(X_valid_smooth))

        elif use_DoubleScalers:
            self._train_DoubleScalers(wave_grid, X_train, y_train)
            X_train = self.scaler_X.forward(torch.FloatTensor(X_train))
            y_train = self.scaler_y.forward(torch.FloatTensor(y_train))
            X_valid = self.scaler_X.forward(torch.FloatTensor(X_valid))
            y_valid = self.scaler_y.forward(torch.FloatTensor(y_valid))

        else:
            # no QSOScaler preprocessing here yet
            X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
            X_valid, y_valid = torch.FloatTensor(X_valid), torch.FloatTensor(y_valid)

            if smooth:
                X_train_smooth = Variable(torch.FloatTensor(X_train_smooth))
                X_valid_smooth = Variable(torch.FloatTensor(X_valid_smooth))

            self.scaler_X = None
            self.scaler_y = None

        # set the number of batches
        n_batches = len(X_train) // self.batch_size

        # set up the arrays for storing and checking the loss
        running_loss = np.zeros(self.num_epochs)
        valid_loss = np.zeros(self.num_epochs)
        min_valid_loss = np.inf

        # train the model to find good residuals
        for epoch in range(self.num_epochs):
            # shuffle training data
            if smooth:
                X_train_new, y_train_new, X_train_smooth_new = shuffle(X_train, y_train, X_train_smooth)
            else:
                X_train_new, y_train_new = shuffle(X_train, y_train)

            # train in batches
            for i in range(n_batches):
                start = i * self.batch_size
                end = start + self.batch_size
                #inputs_np = X_train_new[start:end].numpy()
                inputs = Variable(X_train_new[start:end])
                targets = Variable(y_train_new[start:end])

                if smooth:
                    inputs_smooth = X_train_smooth_new[start:end]
                #targets_np = y_train_new[start:end].numpy()
                #inputs = Variable(torch.FloatTensor(inputs_np))

                # compute the target residuals
                #target_resids = get_rel_resids(inputs, targets)
                #target_resids = get_rel_resids(inputs_np, targets_np)
                #target_resids = Variable(torch.FloatTensor(target_resids))

                # set gradients to zero
                self.optimizer.zero_grad()

                # forward
                if smooth:
                    outputs = self.net(inputs, inputs_smooth)
                else:
                    outputs = self.net(inputs)
                #output_resids = get_rel_resids(inputs, outputs)
                #output_resids = get_rel_resids(inputs_np, outputs.detach().numpy())
                #output_resids = Variable(torch.FloatTensor(output_resids))

                # backward
                outputs_real = self.scaler_y.backward(outputs)
                targets_real = self.scaler_y.backward(targets)
                loss = self.criterion(outputs_real, targets_real)
                #loss = self.criterion(output_resids, target_resids)
                #loss = Variable(loss, requires_grad=True)   # this may not be necessary
                #print("Training loss (mini-batch): {:12.3f}".format(loss))
                loss.backward()

                # optimize
                self.optimizer.step()

                running_loss[epoch] += loss.item()

            print("Epoch " + str(epoch+1) + "/" + str(self.num_epochs) + "completed.")

            # now use the validation set
            if smooth:
                X_valid_new, y_valid_new, X_valid_smooth_new = shuffle(X_valid, y_valid, X_valid_smooth)
            else:
                X_valid_new, y_valid_new = shuffle(X_valid, y_valid)
            validinputs = Variable(torch.FloatTensor(X_valid_new.numpy()))
            validtargets = Variable(torch.FloatTensor(y_valid_new.numpy()))
            #valid_target_resids = get_rel_resids(validinputs, validtargets)

            # compute the validation set output residuals
            if smooth:
                validoutputs = self.net(validinputs, X_valid_smooth_new)
            else:
                validoutputs = self.net(validinputs)
            #valid_output_resids = get_rel_resids(validinputs, validoutputs)

            # compute the loss
            validoutputs_real = self.scaler_y.backward(validoutputs)
            validtargets_real = self.scaler_y.backward(validtargets)
            validlossfunc = self.criterion(validoutputs_real, validtargets_real)
            #validlossfunc = self.criterion(valid_output_resids, valid_target_resids)
            valid_loss[epoch] = validlossfunc.item()
            print("Validation loss: {:12.3f}".format(valid_loss[epoch]/len(X_valid)))

            # save the model if the validation loss decreases
            if min_valid_loss > valid_loss[epoch]:
                print("Validation loss decreased.")
                min_valid_loss = valid_loss[epoch]

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "valid_loss": valid_loss[epoch],
                }, savefile)

        # compute the loss per quasar
        running_loss = running_loss / len(X_train)
        valid_loss = valid_loss / len(X_valid)

        # load the model with lowest validation loss
        checkpoint = torch.load(savefile)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        print ("Best epoch:", checkpoint["epoch"])

        # save the diagnostics in the object
        self.training_loss = running_loss
        self.valid_loss = valid_loss


class DoubleScalingTrainer(Trainer):
    def __init__(self, net, optimizer, criterion, batch_size=1000, num_epochs=400):
        super(DoubleScalingTrainer, self).__init__(net, optimizer, criterion,\
                                          batch_size=batch_size, num_epochs=num_epochs)

    def _train_glob_scalers(self, trainset,\
                            smoothwindow=20, floorval=0.05):
        '''Trains the global QSOScaler on the locally scaled training set.'''

        # first do the local transformation
        wave_grid = trainset.wave_grid
        flux = trainset.flux
        flux_smooth = trainset.flux_smooth
        cont = trainset.cont

        loc_scaler_train = SmoothScaler(wave_grid, flux_smooth)
        flux_train_locscaled = loc_scaler_train.forward(torch.FloatTensor(flux))
        cont_train_locscaled = loc_scaler_train.forward(torch.FloatTensor(cont))

        # now train the global scaler
        flux_train_locscaled_np = flux_train_locscaled.detach().numpy()
        cont_train_locscaled_np = cont_train_locscaled.detach().numpy()

        flux_mean = np.mean(flux_train_locscaled_np, axis=0)
        flux_std = np.std(flux_train_locscaled_np, axis=0) + floorval * np.median(flux_mean)
        cont_mean = np.mean(cont_train_locscaled_np, axis=0)
        cont_std = np.std(cont_train_locscaled_np, axis=0) + floorval * np.median(cont_mean)

        self.glob_scaler_flux = QuasarScaler(wave_grid, flux_mean, flux_std)
        self.glob_scaler_cont = QuasarScaler(wave_grid, cont_mean, cont_std)


    def train_unet(self, trainset, validset, savefile="LinearUNet.pth"):
        '''Train the network.'''

        # train the global scalers
        self._train_glob_scalers(trainset)

        wave_grid = trainset.wave_grid

        # set the number of batches
        n_batches = len(trainset.flux) // self.batch_size

        # set up the arrays for storing and checking the loss
        running_loss = np.zeros(self.num_epochs)
        valid_loss = np.zeros(self.num_epochs)
        min_valid_loss = np.inf

        # set up DataLoaders for the training and validation set
        train_loader = DataLoader(trainset, batch_size=self.batch_size,\
                                  shuffle=True)
        valid_loader = DataLoader(validset, batch_size=self.batch_size,\
                                  shuffle=True)

        # now do mini-batch learning
        for epoch in range(self.num_epochs):

            for flux_train, flux_smooth_train, cont_train in train_loader:

                # set up the local scaler for this batch
                loc_scaler = SmoothScaler(wave_grid, flux_smooth_train)

                # doubly transform the batch input spectra
                flux_train_scaled = loc_scaler.forward(flux_train)
                flux_train_scaled = self.glob_scaler_flux.forward(flux_train_scaled)
                flux_train_scaled = Variable(torch.FloatTensor(flux_train_scaled.numpy()))

                # set gradients to zero
                self.optimizer.zero_grad()

                # forward the network
                outputs = self.net(flux_train_scaled)

                # backward
                # compute loss in physical space
                outputs = self.glob_scaler_cont.backward(outputs)
                outputs_real = loc_scaler.backward(outputs)

                loss = self.criterion(outputs_real, torch.FloatTensor(cont_train.numpy()))
                loss.backward()

                # optimize
                self.optimizer.step()

                running_loss[epoch] += loss.item()

            print("Epoch " + str(epoch + 1) + "/" + str(self.num_epochs) + "completed.")

            # now use the validation set
            for flux_valid, flux_smooth_valid, cont_valid in valid_loader:

                loc_scaler_valid = SmoothScaler(wave_grid, flux_smooth_valid)
                flux_valid_scaled = loc_scaler_valid.forward(flux_valid)
                flux_valid_scaled = self.glob_scaler_flux.forward(flux_valid_scaled)
                flux_valid_scaled = Variable(torch.FloatTensor(flux_valid_scaled.numpy()))

                validoutputs = self.net(flux_valid_scaled)
                validoutputs = self.glob_scaler_cont.backward(validoutputs)
                validoutputs_real = loc_scaler_valid.backward(validoutputs)

                validlossfunc = self.criterion(validoutputs_real, torch.FloatTensor(cont_valid.numpy()))
                valid_loss[epoch] += validlossfunc.item()

            print("Validation loss: {:12.3f}".format(valid_loss[epoch] / len(validset.flux)))

            # save the model if the validation loss decreases
            if min_valid_loss > valid_loss[epoch]:
                print("Validation loss decreased.")
                min_valid_loss = valid_loss[epoch]

                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "valid_loss": valid_loss[epoch],
                }, savefile)

        # compute the loss per quasar
        running_loss = running_loss / len(trainset.flux)
        valid_loss = valid_loss / len(validset.flux)

        # load the model with lowest validation loss
        checkpoint = torch.load(savefile)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        print("Best epoch:", checkpoint["epoch"])

        # save the diagnostics in the object
        self.training_loss = running_loss
        self.valid_loss = valid_loss