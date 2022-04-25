import torch

class MedianScaler:
    def __init__(self, mean_spectrum, floorval=0.05):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_spectrum = torch.tensor(mean_spectrum).float().to(self.device)

        self.median = torch.median(self.mean_spectrum) + floorval

    def forward(self, qso_spectrum):

        try:
            spec_scaled = (qso_spectrum.to(self.device) - self.mean_spectrum) / self.median

        except:
            spec_scaled = (qso_spectrum.to(self.device) - self.mean_spectrum.to(self.device)) / self.median.to(self.device)

        return spec_scaled

    def backward(self, Y):

        try:
            spec = Y.to(self.device) * self.median + self.mean_spectrum

        except:
            spec = Y.to(self.device) * self.median.to(self.device) + self.mean_spectrum.to(self.device)

        return spec