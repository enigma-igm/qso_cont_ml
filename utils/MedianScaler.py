import torch
from IPython import embed

class MedianScaler:
    """
    A class for scaling spectra by taking their difference w.r.t. a wavelength-specific mean and dividing by the median
    of this mean spectrum.

    Attributes:
        device: torch device instance
        mean_spectrum: torch tensor of shape (n_wav,) or (n_wav,n_channels)
        median: torch tensor of shape (1,)

    Methods:
        forward(qso_spectrum)
            Scale a set of spectra.
        backward(Y)
            Descale a set of scaled spectra.
        updateDevice()
            Update the device.
    """

    def __init__(self, mean_spectrum, floorval=0.05):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_spectrum = torch.tensor(mean_spectrum).float().to(self.device)

        median = torch.median(self.mean_spectrum, dim=-1)

        # have to add the floor value row-wise
        try:
            self.median = torch.zeros_like(self.mean_spectrum)
            for i in range(len(median)):
                self.median[i,:] = median[0][i] + floorval
        except:
            self.median = median + floorval


    def forward(self, qso_spectrum):

        try:

            try:
                if qso_spectrum.shape[1] == self.mean_spectrum.shape[0]:
                    spec_scaled = (qso_spectrum.to(self.device) - self.mean_spectrum) / self.median

                else:
                    spec_scaled = (qso_spectrum.to(self.device) - self.mean_spectrum[0]) / self.median[0]

            except:
                if qso_spectrum.shape[1] == self.mean_spectrum.shape[0]:
                    spec_scaled = (qso_spectrum.to(self.device) - self.mean_spectrum.to(self.device)) / self.median.to(self.device)

                else:
                    spec_scaled = (qso_spectrum.to(self.device) - self.mean_spectrum[0].to(self.device)) / self.median[0].to(self.device)

        except:
            print ("Input shape:", qso_spectrum.shape)
            embed()

        return spec_scaled

    def backward(self, Y):

        try:
            spec = Y.to(self.device) * self.median + self.mean_spectrum

        except:
            spec = Y.to(self.device) * self.median.to(self.device) + self.mean_spectrum.to(self.device)

        return spec

    def updateDevice(self):
        """
        Update the device on which the scaler and its attributes are loaded.
        Call this method to use scaler that was initialised on GPU but now has to work on CPU (or vice versa).

        @return:
            self.device: torch.device instance
                The found device on which the scaler's attributes are now loaded.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_spectrum = self.mean_spectrum.to(self.device)
        self.median = self.median.to(self.device)

        return self.device