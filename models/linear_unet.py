# contains the model class for U-Net-like structures
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from pypeit.utils import fast_running_median


class Operator:
    def __init__(self, operator="addition"):
        self.name = operator

    def __call__(self, a, b):

        if self.name=="addition":
            return a+b
        elif self.name=="multiplication":
            return a*b

def get_rel_resids(flux, cont):

    Delta = cont/flux - 1
    return Delta


def get_activfunc(activfunc="relu"):
    if activfunc == "relu":
        activ = nn.ReLU()
    elif activfunc == "elu":
        activ = nn.ELU()
    elif activfunc == "sigmoid":
        activ = nn.Sigmoid()

    return activ


class LinearDownBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, size_hidden, activfunc="relu"):
        super(LinearDownBlock, self).__init__()
        self.activ = get_activfunc(activfunc=activfunc)

        l1 = [in_dim] + size_hidden
        l2 = size_hidden + [out_dim]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1,h2), self.activ]
        layers.pop()
        self.block = nn.Sequential(*layers)

    def forward(self, x):

        res = self.block(x)
        return res


class LinearUpBlock(torch.nn.Module):
    # not what we want
    def __init__(self, in_dim, out_dim, size_hidden, res_down, activfunc="relu"):
        super(LinearUpBlock, self).__init__()
        self.activ = get_activfunc(activfunc=activfunc)
        self.res_down = res_down

        l1 = [in_dim] + size_hidden
        l2 = size_hidden + [out_dim]
        layers = []
        for h1, h2 in zip(l1, l2):
            layers += [nn.Linear(h1, h2), self.activ]
        layers.pop()
        self.block = nn.Sequential(*layers)


class LinearUNet(torch.nn.Module):
    def __init__(self, in_out_dim, size_hidden, activfunc="relu", operator="addition"):
        '''Linear additive U-Net-like model where the input and output have the
        same dimensions.'''

        super(LinearUNet, self).__init__()

        self.downlayer1 = nn.Linear(in_out_dim, size_hidden[0])
        self.downlayer2 = nn.Linear(size_hidden[0], size_hidden[1])
        self.downlayer3 = nn.Linear(size_hidden[1], size_hidden[2])
        self.uplayer1 = nn.Linear(size_hidden[2], size_hidden[1])
        self.uplayer2 = nn.Linear(size_hidden[1], size_hidden[0])
        self.uplayer3 = nn.Linear(size_hidden[0], in_out_dim)

        #self.predict = nn.Linear(size_hidden, in_out_dim)

        self.activ = get_activfunc(activfunc=activfunc)
        self.operator = Operator(operator=operator)
        #downblock = LinearDownBlock(in_out_dim, size_hidden_down, activfunc=activfunc)
        # set up the up block


    def forward(self, x):
        # first perform the transformation to the latent space (dim size_hidden)
        # then apply the activation function to the result
        Ydown1 = self.activ(self.downlayer1(x))

        # do the same for the second layer
        Ydown2 = self.activ(self.downlayer2(Ydown1))
        Ydown3 = self.activ(self.downlayer3(Ydown2))

        # now we go up
        # add Ydown2 to the result before applying the activation function
        Yup1 = self.uplayer1(Ydown3)
        Yup1 = self.operator(Yup1, Ydown2)
        Yup1 = self.activ(Yup1)

        # do the same thing for the second up layer
        Yup2 = self.uplayer2(Yup1)
        Yup2 = self.operator(Yup2, Ydown1)
        Yup2 = self.activ(Yup2)

        Yup3 = self.uplayer3(Yup2)
        # smooth the input x before adding it
        #x_smooth = fast_running_median(x, 20)   # gives rise to an error
        #result = self.operator(Yup3, x_smooth)
        result = self.operator(Yup3, x)
        # result = self.activ(Yup3)   # not necessary --> forces positives

        return result


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