#!/usr/bin/env python

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt


class Image:
    def __init__(self, filename):
        """
        
        :param float charac_rad:
        :param string filename: path+name of the file
        :param string type_im: MUSE, HST etc ... 
        """

        self.header = fits.getheader(filename)
        self.data = np.nan_to_num(fits.getdata(filename))
        self.high = self.header['NAXIS1']
        self.length = self.header['NAXIS2']
        self.size = np.array(np.shape(self.data))
        self.len = self.length*self.high
        self.oversample = None
        self.mask = self.data != 0


class ImageOverSamp(Image):
    def __init__(self, filename, charac_rad):
        Image.__init__(self, filename)

        self.oversample = int(np.ceil(8 / charac_rad))

        x = np.linspace(0, self.length, self.length)
        y = np.linspace(0, self.high, self.high)
        new_x = np.linspace(0, self.length, self.length * int(self.oversample))
        new_y = np.linspace(0, self.high, self.high * int(self.oversample))
        data = np.reshape(self.data, -1)

        if self.oversample > 1:

            func = interp2d(x, y, data, kind='cubic', fill_value=0)
            self.data = np.array(func(new_x, new_y)).transpose()

            self.high *= self.oversample
            self.length *= self.oversample
            self.size *= self.oversample
