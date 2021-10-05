import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.utils import shuffle
import numpy as np
from qso_fitting.models.utils.QuasarScaler import QuasarScaler
from models.network import normalise, rescale_backward, Net
from utils.errorfuncs import MSE, corr_matrix_relresids

def create_learners(parameters, learning_rate=0.1):
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    criterion = torch.nn.MSELoss(size_average=False)

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


def train_model(wave_grid, X_train, y_train, X_valid, y_valid, net,\
                optimizer, criterion, batch_size=1000,\
                num_epochs=1000, learning_rate=0.1, size_hidden=100):
    '''Currently does not use a validation set!'''

    # first train the scalers
    scaler_X, scaler_y = train_scalers(wave_grid, X_train, y_train)

    # use the scalers to normalise input and target
    X_train_normed = normalise(scaler_X, X_train)
    y_train_normed = normalise(scaler_y, y_train)

    batch_no = len(X_train) // batch_size
    #n_feature = X_train.shape[1]
    #n_output = y_train.shape[1]

    running_loss = np.zeros(num_epochs)
    # also calculate the MSE on the rescaled output spectra for the training set
    #training_loss = np.zeros(num_epochs)

    # setup the validation set for computing the MSE
    X_valid_normed = normalise(scaler_X, X_valid)
    y_valid_normed = normalise(scaler_y, y_valid)
    input_valid = Variable(torch.FloatTensor(X_valid_normed.numpy()))
    labels_valid = Variable(torch.FloatTensor(y_valid_normed.numpy()))
    mse_loss_valid = np.zeros(num_epochs)
    # setup a validation loss criterion
    # save the model parameters if the validation loss decreases
    min_valid_loss = np.inf

    for epoch in range(num_epochs):
        X_train_new, y_train_new = shuffle(X_train_normed, y_train_normed)

        for i in range(batch_no):
            start = i * batch_size
            end = start + batch_size
            inputs = Variable(torch.FloatTensor(X_train_new[start:end].numpy()))
            labels = Variable(torch.FloatTensor(y_train_new[start:end].numpy()))

            # set gradients to zero
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss[epoch] += loss.item()

        print ("Epoch "+str(epoch+1)+"/"+str(num_epochs)+" completed.")
        #print ("Loss: "+str(running_loss[epoch]))

        # now use the validation set
        valid_loss = 0.0
        X_valid_new, y_valid_new = shuffle(X_valid_normed, y_valid_normed)
        validinputs = Variable(torch.FloatTensor(X_valid_new.numpy()))
        validlabels = Variable(torch.FloatTensor(y_valid_new.numpy()))
        validoutputs = net(validinputs)
        validlossfunc = criterion(validoutputs, validlabels)
        #valid_loss = validlossfunc.item() * validinputs.size(0)
        #if not epoch%100:
        #    print (validinputs.size(0))
        valid_loss += validlossfunc.item()
        mse_loss_valid[epoch] = valid_loss

        #output_valid = net(input_valid)
        #mse_loss_valid[epoch] = MSE(labels_valid.detach().numpy(), output_valid.detach().numpy())

        # save the model if the validation loss decreases
        if min_valid_loss > valid_loss:
            print ("Validation loss decreased.")
            min_valid_loss = valid_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "valid_loss": valid_loss
            }, "saved_model.pth")

    # divide the loss arrays by the lengths of the data sets to be able to compare
    running_loss = running_loss/len(X_train)
    mse_loss_valid = mse_loss_valid/len(X_valid)

    # after completing the training route, load the model with lowest validation loss
    bestmodel = Net(len(X_train[1]), 100, len(y_train[1]))
    checkpoint = torch.load("saved_model.pth")
    #bestmodel.load_state_dict(checkpoint["model_state_dict"])
    net.load_state_dict(checkpoint["model_state_dict"])   # this should update net
    print ("Best epoch:", checkpoint["epoch"])
    #net.eval()

    return running_loss, mse_loss_valid, scaler_X, scaler_y


def test_model(X_test, y_test, scaler_X, scaler_y, net):
    '''Test the trained model by determining the MSE on the full model,
    i.e. including the normalisation and back-conversion.'''

    res_test = net.full_predict(X_test, scaler_X, scaler_y)
    mse = MSE(y_test, res_test)
    corr_matrix = corr_matrix_relresids(y_test, res_test, len(y_test))

    return mse, corr_matrix