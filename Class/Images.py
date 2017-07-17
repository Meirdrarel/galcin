import numpy as np
from astropy.io import fits
from scipy.interpolate import interp2d
import Tools.tools as tools


class Image:
    def __init__(self, filename, mask=None):
        """
        Stock fits file and determine all parameters like size etc ...

        :param string filename: path+name of the file, can be a numpy array.
        """
        self.data_rebin = None

        if type(filename) is str:
            self.header = fits.getheader(filename)
            self.data = fits.getdata(filename)
            self.high = self.header['NAXIS1']
            self.length = self.header['NAXIS2']
            self.size = np.array(np.shape(self.data))
            self.len = self.length*self.high
            self.oversample = 1
            if mask is None:
                self.mask = np.logical_not(np.isnan(self.data))
            else:
                self.mask = mask
            self.nan_to_num()

        # Add for create model without Image's class
        if type(filename) is np.ndarray:
            self.header = None
            self.data = np.nan_to_num(filename)
            self.size = np.array(np.shape(self.data))
            self.high = self.size[0]
            self.length = self.size[1]
            self.len = self.length*self.high
            self.oversample = 1
            if mask is None:
                self.mask = np.logical_not(np.isnan(self.data))
            else:
                self.mask = mask
            self.nan_to_num()

    def nan_to_num(self):
        self.data = np.nan_to_num(self.data)

    def get_size(self):
        return self.size

    def get_lenght(self):
        return self.length

    def get_high(self):
        return self.high

    def set_oversamp(self, oversamp):
        self.oversample = oversamp

    def get_oversamp(self):
        return self.oversample

    def interpolation(self):
        x = np.linspace(0, self.length, self.length)  # + 0.5*self.oversample
        y = np.linspace(0, self.high, self.high)  # + 0.5*self.oversample
        new_x = np.linspace(0, self.length, self.length * int(self.oversample))
        new_y = np.linspace(0, self.high, self.high * int(self.oversample))
        data = np.reshape(self.data, -1)

        func = interp2d(x, y, data, kind='linear', fill_value=0)
        self.data = np.array(func(new_x, new_y)).transpose()
        self.high *= self.oversample
        self.length *= self.oversample
        self.size *= self.oversample

    def conv_inter_flux(self, psf):
        self.data_rebin = tools.rebin_data(psf.convolution(self.data), self.oversample)
