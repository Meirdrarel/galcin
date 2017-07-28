import numpy as np
from astropy.io import fits
from scipy.interpolate import interp2d
from Tools import calculus
import logging

logger = logging.getLogger('__galcin__')


class Image:
    """
            Stock fits file and determine all parameters like size etc ...

        :param string or ndarray filename: path+name of the file, can be a numpy array.
        """
    def __init__(self, filename, mask=None):
        self.data_rebin = None

        if type(filename) is str:
            self.header = fits.getheader(filename)
            self.data = fits.getdata(filename)
            self.high = self.header['NAXIS1']
            self.length = self.header['NAXIS2']
            self.size = np.array(np.shape(self.data))
            self.len = self.length*self.high
            self.pix_size_1 = 0
            self.pix_size_2 = 0
            self.calc_pix_size()
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
        """
            convert 'nan' into 0
        """
        self.data = np.nan_to_num(self.data)

    def calc_pix_size(self):
        try:
            self.pix_size_1 = np.sqrt(self.header['CD1_1']**2 + self.header['CD1_2'])
            self.pix_size_2 = np.sqrt(self.header['CD2_1']**2 + self.header['CD2_2'])
        except KeyError:
            self.pix_size_1 = self.header['CDELT1']
            self.pix_size_2 = self.header['CDELT2']
        logger.debug('PSF : pixel size in rad: {} (high) and {} (length)'.format(self.pix_size_1, self.pix_size_2))

    def get_pix_size(self):
        return np.array([self.pix_size_1, self.pix_size_2])

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

    def interpolation(self, new_size=None):
        """
            Perform an interpolation of the image at higher resolution and stock the new image

        """
        x = np.linspace(0, self.length, self.length)
        y = np.linspace(0, self.high, self.high)
        if new_size is None:
            new_x = np.linspace(0, self.length, self.length * int(self.oversample))
            new_y = np.linspace(0, self.high, self.high * int(self.oversample))
        else:
            new_x = np.linspace(0, self.length, new_size[0])
            new_y = np.linspace(0, self.high, new_size[1])
        data = np.reshape(self.data, -1)

        func = interp2d(x, y, data, kind='linear', fill_value=0)
        self.data = np.array(func(new_x, new_y)).transpose()
        if new_size is None:
            self.high *= self.oversample
            self.length *= self.oversample
            self.size *= self.oversample
        else:
            self.high, self.length = new_size
            self.size = new_size

    def conv_inter_flux(self, psf):
        """
            Do a convolution of the interpolated image with a psf and stock the result

        :param PSF psf: psf object
        """
        self.data_rebin = calculus.rebin_data(psf.convolution(self.data), self.oversample)
