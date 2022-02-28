import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=10, activfunc="relu",\
                 activparam=1., padding_mode="zeros"):
        super().__init__()
        ksize = (kernel_size,)
        self.conv1 = nn.Conv1d(in_ch, out_ch, ksize, padding_mode=padding_mode)
        self.activ_func = ActivFunc(activfunc, activparam).act
        #self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3)

    def forward(self, x):
        #return self.relu(self.conv2(self.relu(self.conv1(x))))
        return self.activ_func(self.conv1(x))


class Pool:
    def __init__(self, pool="avg", kernel_size=10):
        dict = {
            "avg": nn.AvgPool1d(kernel_size),
            "max": nn.MaxPool1d(kernel_size),
        }

        self.pool = dict[pool]


class ActivFunc:
    def __init__(self, func="relu", hyperparam=1.):
        dict = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(hyperparam),
            "leaky-relu": nn.LeakyReLU(hyperparam),
            "prelu": nn.PReLU(init=hyperparam)
        }

        self.act = dict[func]


class SkipOperator:
    def __init__(self, operation="concatenation"):
        self.name = operation
        self._parameters = nn.ParameterList(None)

    def __call__(self, a, b):

        if self.name=="concatenation":
            return torch.cat([a, b], dim=1)

        elif self.name=="addition":
            return torch.add(a, b)

        elif self.name=="multiplication":
            return torch.mul(a, b)

        elif self.name=="none":
            return a

        else:
            raise ValueError ("Unsupported operator given.")


class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256), kernel_size=10, pool="avg",\
                 pool_kernel_size=10, activfunc="relu", activparam=1.0,\
                 padding_mode="zeros"):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for i in range(len(chs)-1)]

        if isinstance(pool_kernel_size, int):
            pool_kernel_size = [pool_kernel_size for i in range(len(chs)-1)]

        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kernel_size[i],\
                                               activfunc, activparam, padding_mode) for i in range(len(chs)-1)])
        self.pools = nn.ModuleList([Pool(pool, pool_kernel_size[i]).pool for i in range(len(chs)-1)])

    def forward(self, x):
        ftrs = []
        for block, pool in zip(self.enc_blocks, self.pools):
            x = block(x)
            ftrs.append(x)
            x = pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64), kernel_size=10, upconv_kernel_size=1,\
                 activfunc="relu", activparam=1.0, skip="concatenation",\
                 padding_mode="zeros"):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size for i in range(len(chs)-1)]

        if isinstance(upconv_kernel_size, int):
            upconv_kernel_size = [upconv_kernel_size for i in range(len(chs)-1)]

        self.chs = chs
        # only allow concatenation skip connections
        self.upconvs = nn.ModuleList([nn.ConvTranspose1d(chs[i], chs[i+1], \
                                                         upconv_kernel_size[i], (2,)) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kernel_size[i], \
                                               activfunc, activparam, padding_mode) for i in range(len(chs)-1)])

        #if skip=="concatenation:":
        #    self.upconvs = nn.ModuleList([nn.ConvTranspose1d(chs[i], chs[i+1],\
        #                                     upconv_kernel_size[i], (2,)) for i in range(len(chs)-1)])
        #    print ("Skip connection type: concatenation.")
        #    print ("self.upconvs:", self.upconvs)

        #else:
        #    print ("Skip connection type:", skip)
        #    self.upconvs = nn.ModuleList([nn.ConvTranspose1d(chs[i], chs[i],\
        #                                                     upconv_kernel_size[i], (2,)) for i in range(len(chs)-1)])

        #self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kernel_size[i], \
        #                                       activfunc, activparam) for i in range(len(chs)-1)])
        #print ("self.dec_blocks:", self.dec_blocks)
        self.skip = SkipOperator(skip)

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            #print ("Shape of x after up-convolution:", x.shape)
            enc_ftrs = self.crop(encoder_features[i], x)
            #print ("Shape of encoder features after cropping:", enc_ftrs.shape)
            x = torch.cat([x, enc_ftrs], dim=1)
            #print ("Shape of x after concatenation:", x.shape)
            x = self.dec_blocks[i](x)
            # print (x.shape)

            # crop the decoder output to match the dimensions of the encoder output
            # only for edge effects check
            #enc_ftrs = encoder_features[i]
            #_, _, n_wav = enc_ftrs.shape
            #x2d = torch.unsqueeze(x, dim=-1)
            #x2dcropped = torchvision.transforms.CenterCrop([n_wav,1])(x2d)
            #x1dcropped = torch.squeeze(x2dcropped, dim=-1)
            #x = torch.cat([x1dcropped, enc_ftrs], dim=1)
            #x = self.dec_blocks[i](x)

            # try interpolation rather than cropping
            # interpolate the encoder output onto the dimensions of the decoder output
            #_, _, n_wav = x.shape
            #enc_ftrs = F.interpolate(encoder_features[i], n_wav)
            #x = torch.cat([x, enc_ftrs], dim=1)
            #x = self.dec_blocks[i](x)

        return x

    def crop(self, enc_ftrs, x):
        #_, _, n_wav = enc_ftrs.shape
        #x2d = torch.unsqueeze(x, dim=-1)
        #x2dcrop = torchvision.transforms.CenterCrop([n_wav,1])(x2d)
        #x1dcrop = torch.squeeze(x2dcrop, dim=-1)
        #return x1dcrop

        # crop the encoder features to match the dimensions of the decoder output
        _, _, n_wav = x.shape
        enc_ftrs2d = torch.unsqueeze(enc_ftrs, dim=-1)
        enc_ftrs = torchvision.transforms.CenterCrop([n_wav, 1])(enc_ftrs2d)

        enc_ftrs = torch.squeeze(enc_ftrs, dim=-1)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, out_sz, enc_chs=(1,64,128, 256), dec_chs=(256, 128, 64),\
                 kernel_size_enc=10, kernel_size_dec=10, kernel_size_upconv=10,\
                 num_class=1, retain_dim=False, pool="avg", pool_kernel_size=10,\
                 activfunc="relu", activparam=1.0, final_skip=False, skip="concatenation",\
                 padding_mode="zeros"):
        super().__init__()
        self.encoder = Encoder(enc_chs, kernel_size_enc, pool, pool_kernel_size,\
                               activfunc, activparam, padding_mode=padding_mode)
        self.decoder = Decoder(dec_chs, kernel_size_dec, kernel_size_upconv,\
                               activfunc, activparam, padding_mode=padding_mode)
        self.retain_dim = retain_dim
        self.out_sz = out_sz
        self.final_skip = final_skip
        if final_skip:
            self.head = nn.Conv1d(dec_chs[-1]+1, num_class, (1,), padding_mode=padding_mode)
        else:
            self.head = nn.Conv1d(dec_chs[-1], num_class, (1,), padding_mode=padding_mode)

        self.skip_op = SkipOperator(skip)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        if self.final_skip:
            # crop input spectrum to match dimension of decoder output
            _, _, n_wav = out.shape
            x2d = torch.unsqueeze(x, dim=-1)
            x2dcrop = torchvision.transforms.CenterCrop([n_wav,1])(x2d)
            x1dcrop = torch.squeeze(x2dcrop, dim=-1)
            out = torch.cat([out, x1dcrop], dim=1)
            #out = self.skip_op(out, x1dcrop)
            #out = torch.cat([out, x1dcrop], dim=1)
            print ("Shape of final skip connection output:", out.shape)

            # crop decoder output to match dimension of input spectrum
            #_, _, n_wav = x.shape
            #out2d = torch.unsqueeze(out, dim=-1)
            #out2dcrop = torchvision.transforms.CenterCrop([n_wav,1])(out2d)
            #out1dcrop = torch.squeeze(out2dcrop, dim=-1)
            #out = torch.cat([out1dcrop, x], dim=1)

            # interpolate the input spectrum onto the dimensions of the decoder output
            #_, _, n_wav = out.shape
            #x_interp = F.interpolate(x, n_wav)
            #out = torch.cat([out, x_interp], dim=1)

        out = self.head(out)

        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)

        #print ("Shape of final output:", out.shape)
        return out

    def load(self, savefile):
        '''Load a previously trained model saved under <<savefile>>'''

        checkpoint = torch.load(savefile, map_location=torch.device("cpu"))

        self.load_state_dict(checkpoint["model_state_dict"])

        print ("Loaded previously trained model + QuasarScalers.")

        return checkpoint["scaler_flux"], checkpoint["scaler_cont"]
