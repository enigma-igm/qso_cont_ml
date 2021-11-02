# contains the network class for building the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(torch.nn.Module):
    def __init__(self, n_feature, size_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, size_hidden)   # hidden layers
        self.predict = torch.nn.Linear(size_hidden, n_output)   # output layers

    def forward(self, x):
        x_activ = F.elu(self.hidden(x))   # activation function for hidden layers
        x_res = self.predict(x_activ)         # linear output
        return x_res

    def full_predict(self, x_regscale, scaler_X=None, scaler_y=None, smooth=False):
        '''Direct method for predicting the continuum without manually normalising the input
        and rescaling the output again.'''

        if scaler_X is None:
            use_QSOScaler = False
        else:
            use_QSOScaler = True

        if use_QSOScaler:
            x_normed = normalise(scaler_X, x_regscale)
        else:
            x_normed = torch.tensor(x_regscale)

        input = Variable(torch.FloatTensor(x_normed.numpy()))
        res_normed = self(input)

        if use_QSOScaler:
            res = scaler_y.backward(res_normed)
            res_regscale = res.detach().numpy()

        else:
            res_regscale = res_normed.detach().numpy()

        return res_regscale




def normalise(scaler, flux):
    '''Normalise a spectrum given a pre-trained QuasarScaler object.'''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flux_tensor = torch.tensor(flux).to(device)
    normalised_flux = scaler.forward(flux_tensor)

    return normalised_flux


def rescale_backward(scaler, normalised_flux):
    '''Scale the normalised spectrum back to a regular spectrum using the
    pre-trained QuasarScaler object.'''

    #device = torch.device("cuda" if torch.cude.is_available() else "cpu")
    rescaled_flux = scaler.backward(normalised_flux)

    return rescaled_flux