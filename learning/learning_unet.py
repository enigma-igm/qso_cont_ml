# code for training the LinearUNet model
import torch
from torch.autograd import Variable
from sklearn.utils import shuffle
import numpy as np
from learning.learning import Trainer
from models.linear_unet import get_rel_resids
from models.network import normalise
from pypeit.utils import fast_running_median
from utils.smooth_scaler import *
from qso_fitting.models.utils import QuasarScaler
from data.load_datasets import SynthSpectra
from torch.utils.data import DataLoader
from utils.errorfuncs import WavWeights

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


    def _train_glob_scalers(self, trainset, floorval=0.05, globscalers="both"):
        '''Train the QuasarScalers on the training spectra and training continua.'''

        wave_grid = trainset.wave_grid
        flux = trainset.flux
        cont = trainset.cont

        flux_mean = np.mean(flux, axis=0)
        flux_std = np.std(flux, axis=0) + floorval * np.median(flux_mean)
        cont_mean = np.mean(cont, axis=0)
        cont_std = np.std(cont, axis=0) + floorval * np.median(cont_mean)

        scaler_flux = QuasarScaler(wave_grid, flux_mean, flux_std)
        scaler_cont = QuasarScaler(wave_grid, cont_mean, cont_std)

        if globscalers=="both":
            self.glob_scaler_flux = scaler_flux
            self.glob_scaler_cont = scaler_cont

        elif globscalers=="flux":
            self.glob_scaler_flux = scaler_flux
            self.glob_scaler_cont = scaler_flux

        elif globscalers=="cont":
            self.glob_scaler_flux = scaler_cont
            self.glob_scaler_cont = scaler_cont


    def train(self, trainset, validset, savefile="LinearUNet.pth",\
              use_QSOScalers=False, smooth=False,\
              use_DoubleScalers=False, loss_space="real-rel",\
              globscalers="both", weight=False, weightpower=1):
        '''DoubleScaler training currently does not work properly!'''

        # use the QSOScaler
        if use_QSOScalers:

            self._train_glob_scalers(trainset, globscalers=globscalers)

        elif use_DoubleScalers:
            pass

        else:
            self.glob_scaler_flux = None
            self.glob_scaler_cont = None

        # set up the arrays for storing and checking the loss
        running_loss = np.zeros(self.num_epochs)
        valid_loss = np.zeros(self.num_epochs)
        min_valid_loss = np.inf

        # set up DataLoaders
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(validset, batch_size=len(validset), shuffle=True)

        if weight:
            Weights = WavWeights(trainset.wave_grid, power=weightpower)
            weights_mse = Weights.weights_in_MSE

        self.wave_grid = trainset.wave_grid

        # train the model to find good residuals
        for epoch in range(self.num_epochs):
            for flux_train_raw, flux_smooth_train_raw, cont_train_raw in train_loader:
                if use_QSOScalers:
                    flux_train = self.glob_scaler_flux.forward(flux_train_raw)
                    flux_smooth_train = self.glob_scaler_flux.forward(flux_smooth_train_raw)
                    cont_train = self.glob_scaler_cont.forward(cont_train_raw)

                else:
                    flux_train = flux_train_raw
                    flux_smooth_train = flux_smooth_train_raw
                    cont_train = cont_train_raw

                # set gradients to zero
                self.optimizer.zero_grad()

                # forward
                if smooth:
                    outputs = self.net(flux_train.type(torch.FloatTensor),\
                                       flux_smooth_train.type(torch.FloatTensor))
                else:
                    outputs = self.net(flux_train.type(torch.FloatTensor))

                # backward
                if loss_space=="real-rel":
                    # compute loss in physical flux quantities
                    # relative to true continuum
                    if use_QSOScalers:
                        outputs_real = self.glob_scaler_cont.backward(outputs)
                        outputs_real_rel = (outputs_real / cont_train_raw).type(torch.FloatTensor)
                        cont_train_rel = (cont_train_raw / cont_train_raw).type(torch.FloatTensor)

                    else:
                        outputs_real_rel = (outputs / cont_train).type(torch.FloatTensor)
                        cont_train_rel = (cont_train / cont_train).type(torch.FloatTensor)

                    if weight:
                        loss = self.criterion(outputs_real_rel*weights_mse, cont_train_rel*weights_mse)

                    else:
                        loss = self.criterion(outputs_real_rel, cont_train_rel)

                elif loss_space=="globscaled":
                    # compute the loss in globally scaled space2w
                    loss = self.criterion(outputs, cont_train.type(torch.FloatTensor))

                loss.backward()

                # optimize
                self.optimizer.step()

                running_loss[epoch] += loss.item()

            print("Epoch " + str(epoch+1) + "/" + str(self.num_epochs) + "completed.")

            # now use the validation set

            for flux_valid_raw, flux_smooth_valid_raw, cont_valid_raw in valid_loader:

                if use_QSOScalers:
                    flux_valid = self.glob_scaler_flux.forward(flux_valid_raw)
                    flux_smooth_valid = self.glob_scaler_flux.forward(flux_smooth_valid_raw)
                    cont_valid = self.glob_scaler_cont.forward(cont_valid_raw)

                else:
                    flux_valid = flux_valid_raw
                    flux_smooth_valid = flux_smooth_valid_raw
                    cont_valid = cont_valid_raw

                # forward the network
                if smooth:
                    validoutputs = self.net(flux_valid.type(torch.FloatTensor),\
                                            flux_smooth_valid.type(torch.FloatTensor))
                else:
                    validoutputs = self.net(flux_valid.type(torch.FloatTensor))

                if loss_space=="real-rel":
                    if use_QSOScalers:
                        validoutputs_real = self.glob_scaler_cont.backward(validoutputs)
                        validoutputs_real_rel = (validoutputs_real / cont_valid_raw).type(torch.FloatTensor)
                        cont_valid_rel = (cont_valid_raw / cont_valid_raw).type(torch.FloatTensor)

                    else:
                        validoutputs_real_rel = (validoutputs / cont_valid).type(torch.FloatTensor)
                        cont_valid_rel = (cont_valid / cont_valid).type(torch.FloatTensor)

                    if weight:
                        validlossfunc = self.criterion(validoutputs_real_rel*weights_mse, cont_valid_rel*weights_mse)

                    else:
                        validlossfunc = self.criterion(validoutputs_real_rel, cont_valid_rel)

                elif loss_space=="globscaled":
                    validlossfunc = self.criterion(validoutputs, cont_valid.type(torch.FloatTensor))

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
                }, savefile)

        # compute the loss per quasar
        running_loss = running_loss / len(trainset)
        #running_loss = running_loss / (trainset.flux.shape[1])
        #valid_loss = valid_loss / (validset.flux.shape[1])

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

    def _train_glob_scalers(self, trainset, floorval=0.05,\
                            globscalers="both", relscaler=True,\
                            relglobscaler=True, abs_descaling=False):
        '''Trains the global QSOScaler on the locally scaled training set.'''

        # first do the local transformation
        wave_grid = trainset.wave_grid
        flux = trainset.flux
        flux_smooth = trainset.flux_smooth
        cont = trainset.cont

        if relscaler:
            loc_scaler_train = SmoothScaler(wave_grid, flux_smooth,\
                                            abs_descaling=abs_descaling)

        else:
            loc_scaler_train = SmoothScalerAbsolute(wave_grid, flux_smooth)

        flux_train_locscaled = loc_scaler_train.forward(torch.FloatTensor(flux))
        cont_train_locscaled = loc_scaler_train.forward(torch.FloatTensor(cont))

        # now train the global scalers
        flux_train_locscaled_np = flux_train_locscaled.detach().numpy()
        cont_train_locscaled_np = cont_train_locscaled.detach().numpy()

        flux_mean = np.mean(flux_train_locscaled_np, axis=0)
        flux_std = np.std(flux_train_locscaled_np, axis=0) + floorval * np.median(flux_mean)
        cont_mean = np.mean(cont_train_locscaled_np, axis=0)
        cont_std = np.std(cont_train_locscaled_np, axis=0) + floorval * np.median(cont_mean)

        if relglobscaler:
            scaler_flux = QuasarScaler(wave_grid, flux_mean, flux_std)
            scaler_cont = QuasarScaler(wave_grid, cont_mean, cont_std)

        else:
            scaler_flux = MeanShiftScaler(wave_grid, flux_mean)
            scaler_cont = MeanShiftScaler(wave_grid, cont_mean)

        if globscalers=="both":
            self.glob_scaler_flux = scaler_flux
            self.glob_scaler_cont = scaler_cont

        elif globscalers=="flux":
            self.glob_scaler_flux = scaler_flux
            self.glob_scaler_cont = scaler_flux

        elif globscalers=="cont":
            self.glob_scaler_flux = scaler_cont
            self.glob_scaler_cont = scaler_cont


    def train_unet(self, trainset, validset, savefile="LinearUNet.pth",\
                   loss_space="real-rel", globscalers="both", relscaler=True,\
                   weight=False, weightpower=1, floorval=0.05,\
                   relglobscaler=True, abs_descaling=False):
        '''Train the network.'''

        # train the global scalers
        self._train_glob_scalers(trainset, globscalers=globscalers,\
                                 relscaler=relscaler, floorval=floorval,\
                                 relglobscaler=relglobscaler,\
                                 abs_descaling=abs_descaling)

        wave_grid = trainset.wave_grid
        self.wave_grid = wave_grid

        # set up the arrays for storing and checking the loss
        running_loss = np.zeros(self.num_epochs)
        valid_loss = np.zeros(self.num_epochs)
        min_valid_loss = np.inf

        # set up DataLoaders for the training and validation set
        train_loader = DataLoader(trainset, batch_size=self.batch_size,\
                                  shuffle=True)
        valid_loader = DataLoader(validset, batch_size=len(validset),\
                                  shuffle=True)

        # load the weights
        if weight:
            Weights = WavWeights(trainset.wave_grid, power=weightpower)
            weights_mse = Weights.weights_in_MSE


        # now do mini-batch learning
        for epoch in range(self.num_epochs):

            for flux_train, flux_smooth_train, cont_train in train_loader:

                # set up the local scaler for this batch
                loc_scaler = SmoothScaler(wave_grid, flux_smooth_train)

                # doubly transform the batch input spectra
                flux_train_scaled = loc_scaler.forward(flux_train)
                flux_train_scaled = self.glob_scaler_flux.forward(flux_train_scaled)
                flux_train_scaled = flux_train_scaled.type(torch.FloatTensor)
                #flux_train_scaled = Variable(torch.FloatTensor(flux_train_scaled.numpy()))

                # set gradients to zero
                self.optimizer.zero_grad()

                # forward the network
                outputs = self.net(flux_train_scaled)

                # backward

                if loss_space=="real-rel":
                    # compute loss in physical flux quantities
                    # relative to the true continuum!
                    outputs_real = loc_scaler.backward(outputs)
                    outputs_real_rel = (outputs_real / cont_train).type(torch.FloatTensor)
                    cont_train_rel = (cont_train / cont_train).type(torch.FloatTensor)

                    if weight:
                        loss = self.criterion(outputs_real_rel*weights_mse, cont_train_rel*weights_mse)
                    else:
                        loss = self.criterion(outputs_real_rel, cont_train_rel)

                elif loss_space=="real-abs":
                    outputs_real = loc_scaler.backward(outputs)

                    if weight:
                        loss = self.criterion(outputs_real*weights_mse, cont_train.type(torch.FloatTensor)*weights_mse)
                    else:
                        loss = self.criteiron(outputs_real, cont_train.type(torch.FloatTensor))

                elif loss_space=="locscaled":
                    # compute loss in locally scaled space
                    outputs_locscaled = self.glob_scaler_cont.backward(outputs)
                    cont_train_locscaled = loc_scaler.forward(cont_train)

                    if weight:
                        loss = self.criterion(outputs_locscaled*weights_mse, cont_train_locscaled.type(torch.FloatTensor)*weights_mse)
                    else:
                        loss = self.criterion(outputs_locscaled, cont_train_locscaled.type(torch.FloatTensor))

                elif loss_space=="locscaled-rel":
                    outputs_locscaled = self.glob_scaler_cont.backward(outputs)
                    cont_train_locscaled = loc_scaler.forward(cont_train)
                    outputs_locscaled_rel = (outputs_locscaled / cont_train_locscaled).type(torch.FloatTensor)
                    cont_train_locscaled_rel = torch.FloatTensor(np.ones(cont_train_locscaled.shape))

                    loss = self.criterion(outputs_locscaled_rel, cont_train_locscaled_rel)

                elif loss_space=="doublyscaled":
                    cont_train_locscaled = loc_scaler.forward(cont_train)
                    cont_train_doubscaled = self.glob_scaler_cont.forward(cont_train_locscaled)
                    loss = self.criterion(outputs, cont_train_doubscaled.type(torch.FloatTensor))

                elif loss_space=="doublyscaled-rel":
                    cont_train_locscaled = loc_scaler.forward(cont_train)
                    cont_train_doubscaled = self.glob_scaler_cont.forward(cont_train_locscaled)
                    outputs_rel = (outputs / cont_train_doubscaled).type(torch.FloatTensor)
                    cont_train_doubscaled_rel = torch.FloatTensor(np.ones(cont_train_doubscaled.shape))
                    loss = self.criterion(outputs_rel, cont_train_doubscaled_rel)

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
                #flux_valid_scaled = Variable(torch.FloatTensor(flux_valid_scaled.numpy()))
                flux_valid_scaled = flux_valid_scaled.type(torch.FloatTensor)

                validoutputs = self.net(flux_valid_scaled)
                validoutputs_locscaled = self.glob_scaler_cont.backward(validoutputs)
                validoutputs_real = loc_scaler_valid.backward(validoutputs_locscaled)

                if loss_space=="real-rel":
                    # compute the loss in real space
                    # normalised by the true continuum for unbiased learning
                    validoutputs_real_rel = (validoutputs_real / cont_valid).type(torch.FloatTensor)
                    cont_valid_rel = (cont_valid / cont_valid).type(torch.FloatTensor)

                    if weight:
                        validlossfunc = self.criterion(validoutputs_real_rel*weights_mse, cont_valid_rel*weights_mse)
                    else:
                        validlossfunc = self.criterion(validoutputs_real_rel, cont_valid_rel)

                elif loss_space=="real-abs":
                    if weight:
                        validlossfunc = self.criterion(validoutputs_real*weights_mse, cont_valid.type(torch.FloatTensor)*weights_mse)
                    else:
                        validlossfunc = self.criterion(validoutputs_real, cont_valid.type(torch.FloatTensor))

                elif loss_space=="locscaled":
                    # compute the loss in locally scaled space
                    cont_valid_locscaled = loc_scaler_valid.forward(cont_valid)

                    if weight:
                        validlossfunc = self.criterion(validoutputs_locscaled*weights_mse, cont_valid_locscaled.type(torch.FloatTensor)*weights_mse)
                    else:
                        validlossfunc = self.criterion(validoutputs_locscaled, cont_valid_locscaled.type(torch.FloatTensor))

                elif loss_space=="locscaled-rel":
                    cont_valid_locscaled = loc_scaler_valid.forward(cont_valid)
                    validoutputs_locscaled_rel = (validoutputs_locscaled / cont_valid_locscaled).type(torch.FloatTensor)
                    cont_valid_locscaled_rel = torch.FloatTensor(np.ones(cont_valid_locscaled.shape))
                    validlossfunc = self.criterion(validoutputs_locscaled_rel, cont_valid_locscaled_rel)

                elif loss_space=="doublyscaled":
                    cont_valid_locscaled = loc_scaler_valid.forward(cont_valid)
                    cont_valid_doubscaled = self.glob_scaler_cont.forward(cont_valid_locscaled)
                    validlossfunc = self.criterion(validoutputs, cont_valid_doubscaled)

                elif loss_space=="doublyscaled-rel":
                    cont_valid_locscaled = loc_scaler_valid.forward(cont_valid)
                    cont_valid_doubscaled = self.glob_scaler_cont.forward(cont_valid_locscaled)
                    validoutputs_rel = (validoutputs / cont_valid_doubscaled).type(torch.FloatTensor)
                    cont_valid_doubscaled_rel = torch.FloatTensor(np.ones(cont_valid_doubscaled.shape))
                    validlossfunc = self.criterion(validoutputs_rel, cont_valid_doubscaled_rel)

                #validlossfunc = self.criterion(validoutputs_real, torch.FloatTensor(cont_valid.numpy()))
                valid_loss[epoch] += validlossfunc.item()

            # divide total validation loss by the number of quasars to average
            print("Validation loss: {:12.3f}".format(valid_loss[epoch] / len(validset)))

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
        running_loss = running_loss / len(trainset)
        valid_loss = valid_loss / len(validset)

        # load the model with lowest validation loss
        checkpoint = torch.load(savefile)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        print("Best epoch:", checkpoint["epoch"])

        # save the diagnostics in the object
        self.training_loss = running_loss
        self.valid_loss = valid_loss