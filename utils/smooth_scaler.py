import numpy as np
import torch
#from qso_fitting.models.utils.QuasarScaler import QuasarScaler
from utils.QuasarScaler import QuasarScaler

class SmoothScaler:
    def __init__(self, wave_rest, flux_smooth, abs_descaling=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wave_rest = torch.tensor(wave_rest).float().to(self.device)
        self.flux_smooth = torch.tensor(flux_smooth).float().to(self.device)
        self.abs_descaling = abs_descaling

    def forward(self, flux):
        return flux.to(self.device)/self.flux_smooth - 1

    def backward(self, relresid):

        if self.abs_descaling:
            return self.flux_smooth * relresid.to(self.device)
        else:
            return self.flux_smooth*(1 + relresid.to(self.device))


class SmoothScalerAbsolute:
    def __init__(self, wave_rest, flux_smooth):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wave_rest = torch.tensor(wave_rest).float().to(self.device)
        self.flux_smooth = torch.tensor(flux_smooth).float().to(self.device)

    def forward(self, flux):
        return (flux.to(self.device) - self.flux_smooth)

    def backward(self, resid):
        return (resid.to(self.device) + self.flux_smooth)


class MeanShiftScaler:
    '''Essentially an absolute QuasarScaler.'''

    def __init__(self, wave_rest, mean_spectrum):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wave_rest = torch.FloatTensor(wave_rest).to(self.device)
        self.mean_spectrum = torch.FloatTensor(mean_spectrum).to(self.device)

    def forward(self, qso_spectrum):
        return qso_spectrum.to(self.device) - self.mean_spectrum

    def backward(self, Y):
        return Y.to(self.device) + self.mean_spectrum



class DoubleScaler:
    '''Scaler that combines the local SmoothScaler and the global
    QuasarScaler.'''

    def __init__(self, wave_rest, flux_train, smoothwindow=20, floorval=0.05,\
                 cont_train=None):

        from pypeit.utils import fast_running_median

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wave_rest = torch.tensor(wave_rest).float().to(self.device)
        self.flux_smooth = torch.tensor(wave_rest).float().to(self.device)

        flux_smooth = np.zeros(flux_train.shape)
        for i in range(len(flux_train)):
            flux_smooth[i, :] = fast_running_median(flux_train[i], smoothwindow)

        self.locscaler = SmoothScaler(wave_rest, flux_smooth)

        if cont_train is None:
            flux_locscaled = (self.locscaler.forward(torch.FloatTensor(flux_train))).detach().numpy()
        else:
            flux_locscaled = (self.locscaler.forward(torch.FloatTensor(cont_train))).detach().numpy()

        flux_mean = np.mean(flux_locscaled, axis=0)
        flux_std = np.std(flux_locscaled, axis=0) + floorval * np.median(flux_mean)

        self.globscaler = QuasarScaler(wave_rest, flux_mean, flux_std)

    def forward(self, flux):
        return self.globscaler.forward(self.locscaler.forward(flux))

    def backward(self, doubly_scaled_flux):
        return self.locscaler.backward(self.globscaler.backward(doubly_scaled_flux))