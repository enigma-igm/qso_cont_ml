# contains the model class for U-Net-like structures
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from pypeit.utils import fast_running_median


class Operator:
    def __init__(self, operator="addition"):
        self.name = operator
        self._parameters = nn.ParameterList(None)

    def __call__(self, a, b):

        if self.name=="addition":
            return a+b
        elif self.name=="multiplication":
            return a*b
        elif self.name=="relative-addition":
            return (1 + a)*b
        else:
            print ("Warning: unsupported operator given; defaulted to addition.")
            return a+b

    def parameters(self):
        return self._parameters

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
    def __init__(self, in_out_dim, size_hidden, activfunc="relu",\
                 operator="addition", no_final_skip=False):
        '''Linear additive U-Net-like model where the input and output have the
        same dimensions.'''

        super(LinearUNet, self).__init__()

        #self.downlayer1 = nn.Linear(in_out_dim, size_hidden[0])
        #self.downlayer2 = nn.Linear(size_hidden[0], size_hidden[1])
        #self.downlayer3 = nn.Linear(size_hidden[1], size_hidden[2])
        #self.uplayer1 = nn.Linear(size_hidden[2], size_hidden[1])
        #self.uplayer2 = nn.Linear(size_hidden[1], size_hidden[0])
        #self.uplayer3 = nn.Linear(size_hidden[0], in_out_dim)

        #self.predict = nn.Linear(size_hidden, in_out_dim)

        self.activ = get_activfunc(activfunc=activfunc)
        self.operator = Operator(operator=operator)
        self.no_final_skip = no_final_skip

        layers = []
        dims = [in_out_dim] + size_hidden

        self.Ndown = len(size_hidden)

        # create the down block
        for i in range(self.Ndown):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(self.activ)

        # create the up block
        for i in range(self.Ndown):
            layers.append(nn.Linear(dims[-i-1], dims[-i-2]))
            layers.append(self.operator)
            if i != len(size_hidden)-1:
                layers.append(self.activ)

        self.model_layers = layers
        #self.parameters = []
        self.params = nn.ParameterList([])

        for layer in layers:
            self.params.extend(layer.parameters())
#                print ("Parameters:", layers._parameters)
            #self.parameters.append(layer._parameters)

        #self._parameters = nn.ParameterList(self.parameters)

        #downblock = LinearDownBlock(in_out_dim, size_hidden_down, activfunc=activfunc)
        # set up the up block

    def forward(self, x, x_smooth=None, smooth=False):

        # go through every layer in self.model_layers
        # save the result of each down-layer + activation function

        layeroutputs = []
        oper_counter = 1
        layeroutputs.append(self.model_layers[0](x))

        for i in range(1,len(self.model_layers)-1):

            try:
                res = self.model_layers[i](layeroutputs[i-1])

            except:
                #print (i)
                #print (2*self.Ndown-2*oper_counter-1)

                res = self.model_layers[i](layeroutputs[i-1],\
                                           layeroutputs[2*self.Ndown-2*oper_counter-1])
                oper_counter += 1

            layeroutputs.append(res)

        # apply the final skip connection
        if self.no_final_skip:
            return layeroutputs[-1]

        else:
            if x_smooth is None:
                if smooth:
                    # smooth the input x before adding it
                    # we need to do this in a loop
                    # and we need to turn the input into an array
                    x_smooth = np.zeros(x.shape)
                    for i in range(len(x)):
                        x_smooth[i, :] = fast_running_median(x.detach().numpy()[i], 20)
                    # x_smooth = fast_running_median(x, 20)   # gives rise to an error
                    result = self.model_layers[-1](layeroutputs[-1],\
                                                Variable(torch.FloatTensor(x_smooth)))

                else:
                    result = self.model_layers[-1](layeroutputs[-1],\
                                                   x)

            else:
                result = self.model_layers[-1](layeroutputs[-1],\
                                               x_smooth)

            return result


        # first perform the transformation to the latent space (dim size_hidden)
        # then apply the activation function to the result
        #Ydown1 = self.activ(self.downlayer1(x))

        # do the same for the second layer
        #Ydown2 = self.activ(self.downlayer2(Ydown1))
        #Ydown3 = self.activ(self.downlayer3(Ydown2))

        # now we go up
        # add Ydown2 to the result before applying the activation function
        #Yup1 = self.uplayer1(Ydown3)
        #Yup1 = self.operator(Yup1, Ydown2)
        #Yup1 = self.activ(Yup1)

        # do the same thing for the second up layer
        #Yup2 = self.uplayer2(Yup1)
        #Yup2 = self.operator(Yup2, Ydown1)
        #Yup2 = self.activ(Yup2)

        #Yup3 = self.uplayer3(Yup2)

        #if self.no_final_skip:
        #    return Yup3

        #else:
        #    if x_smooth is None:
        #        if smooth:
        #            # smooth the input x before adding it
                    # we need to do this in a loop
                    # and we need to turn the input into an array
        #            x_smooth = np.zeros(x.shape)
        #            for i in range(len(x)):
        #                x_smooth[i,:] = fast_running_median(x.detach().numpy()[i], 20)
                    #x_smooth = fast_running_median(x, 20)   # gives rise to an error
        #            result = self.operator(Yup3, Variable(torch.FloatTensor(x_smooth)))

        #        else: result = self.operator(Yup3, x)
        #    else:
        #        result = self.operator(Yup3, x_smooth)
            #result = self.operator(Yup3, x)
            # result = self.activ(Yup3)   # not necessary --> forces positives

        #    return result


    def full_predict(self, x, scaler_X=None, scaler_y=None, smooth=False, x_smooth=None):

        x = torch.FloatTensor(x)

        if scaler_X is None:
            input = Variable(x)
            if smooth:
                x_smooth = Variable(torch.FloatTensor(x_smooth))
            res = self(input, smooth=smooth, x_smooth=x_smooth)
            res_np = res.detach().numpy()

        else:
            # apply the scaler after smoothing
            if smooth:
                x_smooth_scaled = scaler_X.forward(torch.FloatTensor(x_smooth))
            else:
                x_smooth_scaled = None

            x_scaled = scaler_X.forward(x)
            input = Variable(x_scaled)
            res = self(input, smooth=smooth, x_smooth=x_smooth_scaled)
            res_descaled = scaler_y.backward(res)
            res_np = res_descaled.detach().numpy()

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