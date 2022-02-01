'''Copied QuasarScaler class from qso_fitting module'''

import torch

class QuasarScaler(object):
    def __init__(self, wave_rest, mean_spectrum, std_spectrum):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wave_rest = torch.tensor(wave_rest).float().to(self.device)
        self.mean_spectrum = torch.tensor(mean_spectrum).float().to(self.device)
        self.std_spectrum = torch.tensor(std_spectrum).float().to(self.device)

    def forward(self, qso_spectrum):
        return (qso_spectrum.to(self.device) - self.mean_spectrum) / self.std_spectrum

    def backward(self, Y):
        return Y.to(self.device) * self.std_spectrum + self.mean_spectrum