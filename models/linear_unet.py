# contains the model class for U-Net-like structures
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_rel_resids(flux, cont):

    Delta = cont/flux - 1
    return Delta

class LinearUNet(torch.nn.Module):
    def __init__(self, in_out_dim, size_hidden):
        '''Linear additive U-Net-like model where the input and output have the
        same dimensions.'''

        super(LinearUNet, self).__init__()
        # first try one layer
        self.hidden = nn.Linear(in_out_dim, size_hidden)
        self.predict = nn.Linear(size_hidden, in_out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # first perform the transformation to the latent space (dim size_hidden)
        # then apply the activation function to the result
        x_activ = F.relu(self.hidden(x))

        # now go back to real space (dim in_out_dim)
        x_res = self.predict(x_activ)

        # compute relative residuals in output space
        # we want our training routine to find the right residuals
        # use a sigmioid activation function to force the residuals to be +
        self.rel_resids = self.sigmoid(get_rel_resids(x, x_res))

        return x_res

    def backward(self, x):

        x_flux = x / (self.rel_resids + 1)
        return x_flux


    def full_predict(self, x, scaler_X=None, scaler_y=None):

        x = torch.FloatTensor(x)
        input = Variable(x)
        res = self(input)
        res_np = res.detach().numpy()

        return res_np


class NeuralNet(nn.Module):
    def __init__(self, hidden, in_dim, out_dim):
        super(NeuralNet, self).__init__()
        l1 = [in_dim] + hidden
        l2 = hidden + [out_dim]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), nn.PReLU()]
        layers.pop()
        self.net = nn.Sequential(*layers)