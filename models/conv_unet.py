import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=10):
        super().__init__()
        ksize = (kernel_size,)
        self.conv1 = nn.Conv1d(in_ch, out_ch, ksize)
        self.relu = nn.ReLU()
        #self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3)

    def forward(self, x):
        #return self.relu(self.conv2(self.relu(self.conv1(x))))
        return self.relu(self.conv1(x))


class Encoder(nn.Module):
    def __init__(self, chs=(1,64,128,256), kernel_size=10):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kernel_size) for i in range(len(chs)-1)])
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64), kernel_size=10):
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose1d(chs[i], chs[i+1],\
                                         kernel_size, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            #print ("Shape of x after up-convolution:", x.shape)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
            #print (x.shape)
        return x

    def crop(self, enc_ftrs, x):
        _, _, n_wav = x.shape
        enc_ftrs2d = torch.unsqueeze(enc_ftrs, dim=-1)
        #enc_ftrs2d = torch.FloatTensor(np.expand_dims(enc_ftrs.detach().numpy(),\
        #                                              axis=3))
        enc_ftrs = torchvision.transforms.CenterCrop([n_wav,1])(enc_ftrs2d)
        #print (enc_ftrs.shape)
        #enc_ftrs = np.squeeze(enc_ftrs, axis=-1)

        enc_ftrs = torch.squeeze(enc_ftrs, dim=-1)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(self, out_sz, enc_chs=(1,64,128, 256), dec_chs=(256, 128, 64),\
                 kernel_size_enc=10, kernel_size_dec=10, num_class=1,\
                 retain_dim=False):
        super().__init__()
        self.encoder = Encoder(enc_chs, kernel_size_enc)
        self.decoder = Decoder(dec_chs, kernel_size_dec)
        self.head = nn.Conv1d(dec_chs[-1], num_class, (1,))
        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
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

        #return checkpoint["scaler_flux"], checkpoint["scaler_cont"]
        return checkpoint["valid_loss"]
