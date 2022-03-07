'''File containing the MinMaxScaler class for scaling the spectra in way that preserves
their shape. The spectra are all scaled onto a [0,1] interval.'''

import torch

class MinMaxScaler:
    def __init__(self, minimum, maximum):
        '''Scaling that scales the spectra to lie within the interval [0,1],
        while preserving the spectrum shapes.'''

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.minimum = torch.tensor(minimum).float().to(self.device)
        self.maximum = torch.tensor(maximum).float().to(self.device)

    def forward(self, qso_spectrum):
        try:
            numerator = qso_spectrum.to(self.device) - self.minimum
            denominator = self.maximum - self.minimum

        except:
            numerator = qso_spectrum.to(self.device) - self.minimum.to(self.device)
            denominator = self.maximum.to(self.device) - self.minimum.to(self.device)

        return numerator / denominator

    def backward(self, Y):
        try:
            spec = self.minimum + Y.to(self.device) * (self.maximum - self.minimum)

        except:
            spec = self.minimum.to(self.device) + Y.to(self.device) * (self.maximum.to(self.device) - self.minimum.to(self.device))

        return spec