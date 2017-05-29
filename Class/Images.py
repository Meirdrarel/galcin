import numpy as np
from astropy.io import fits
from scipy.interpolate import interp2d
import Tools.tools as tools


class Image:
    def __init__(self, filename, mask=True):
        """
        
        :param string filename: path+name of the file
        """
        self.data_rebin = None

        if type(filename) is str:
            self.header = fits.getheader(filename)
            self.data = np.nan_to_num(fits.getdata(filename))
            self.high = self.header['NAXIS1']
            self.length = self.header['NAXIS2']
            self.size = np.array(np.shape(self.data))
            self.len = self.length*self.high
            self.oversample = 1
            self.mask = self.data >= 0.1

        # Add for create model without images
        if type(filename) is np.ndarray:
            self.header = None
            self.data = filename
            self.size = np.array(np.shape(self.data))
            self.high = self.size[0]
            self.length = self.size[1]
            self.len = self.length*self.high
            self.oversample = 1
            self.mask = self.data >= 0.1

    def conv_inter_flux(self, psf):
        self.data_rebin = tools.rebin_data(psf.convolution(self.data), self.oversample)


class ImageOverSamp(Image):
    def __init__(self, filename, charac_rad, oversamp=None):
        Image.__init__(self, filename)

        if oversamp:
            self.oversample = oversamp
        else:
            self.oversample = int(np.ceil(8 / charac_rad))  # from idl code

        x = np.linspace(0, self.length, self.length)  # + 0.5*self.oversample
        y = np.linspace(0, self.high, self.high)  # + 0.5*self.oversample
        new_x = np.linspace(0, self.length, self.length * int(self.oversample))
        new_y = np.linspace(0, self.high, self.high * int(self.oversample))
        data = np.reshape(self.data, -1)

        if self.oversample > 1:
            func = interp2d(x, y, data, kind='linear', fill_value=0)
            self.data = np.array(func(new_x, new_y)).transpose()
            self.high *= self.oversample
            self.length *= self.oversample
            self.size *= self.oversample
