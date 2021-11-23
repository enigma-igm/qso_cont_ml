# code for training the LinearUNet model
import torch
from torch.autograd import Variable
from sklearn.utils import shuffle
import numpy as np
from learning.learning import Trainer
from models.linear_unet import get_rel_resids
from models.network import normalise
from pypeit.utils import fast_running_median

class UNetTrainer(Trainer):
    def __init__(self, net, optimizer, criterion, batch_size=1000, num_epochs=400):
        super(UNetTrainer, self).__init__(net, optimizer, criterion, batch_size=batch_size, num_epochs=num_epochs)

    def train(self, wave_grid, X_train, y_train, X_valid, y_valid,\
              savefile="LinearUNet.pth", use_QSOScalers=False, smooth=False):

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

        else:
            # no QSOScaler preprocessing here yet
            X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
            X_valid, y_valid = torch.FloatTensor(X_valid), torch.FloatTensor(y_valid)

            if smooth:
                X_train_smooth = Variable(torch.FloatTensor(X_train_smooth))
                X_valid_smooth = Variable(torch.FloatTensor(X_valid_smooth))

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
                loss = self.criterion(outputs, targets)
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
            validlossfunc = self.criterion(validoutputs, validtargets)
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