import numpy as np
import torch
from qso_fitting.models.utils.QuasarScaler import QuasarScaler
from pypeit.utils import fast_running_median

class SmoothScaler:
    def __init__(self, wave_rest, flux_smooth):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wave_rest = torch.tensor(wave_rest).float().to(self.device)
        self.flux_smooth = torch.tensor(flux_smooth).float().to(self.device)

    def forward(self, flux):
        return flux.to(self.device)/self.flux_smooth - 1

    def backward(self, relresid):
        return self.flux_smooth*(1 + relresid.to(self.device))


class DoubleScaler:
    '''Scaler that combines the local SmoothScaler and the global
    QuasarScaler.'''

    def __init__(self, wave_rest, flux_train, smoothwindow=20, floorval=0.05,\
                 cont_train=None):
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