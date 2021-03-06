'''Copied QuasarScaler class from qso_fitting module'''

import torch

class QuasarScaler(object):
    def __init__(self, wave_rest, mean_spectrum, std_spectrum):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wave_rest = torch.tensor(wave_rest).float().to(self.device)
        self.mean_spectrum = torch.tensor(mean_spectrum).float().to(self.device)
        self.std_spectrum = torch.tensor(std_spectrum).float().to(self.device)

    def forward(self, qso_spectrum):
        try:
            spec_scaled = (qso_spectrum.to(self.device) - self.mean_spectrum) / self.std_spectrum
        except:
            spec_scaled = (qso_spectrum.to(self.device) - self.mean_spectrum.to(self.device)) / self.std_spectrum.to(self.device)

        return spec_scaled

    def backward(self, Y):
        try:
            spec = Y.to(self.device) * self.std_spectrum + self.mean_spectrum
        except:
            spec = Y.to(self.device) * self.std_spectrum.to(self.device) + self.mean_spectrum.to(self.device)

        return spec