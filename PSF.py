import math
import numpy as np
from numpy.fft import fftshift, fft2, ifft2
from Images import Image


class PSF:

    def __init__(self, flux_hd, img_psf=None, fwhm=3):
        """
        If psf image isn't given, a gaussian is used

        :param Image flux_hd:
        :param ndarray img_psf: image of the psf
        :param float fwhm: fwhm of a gaussian, default is 2.5 pixel
        """

        self.psf = img_psf
        self.im_size = flux_hd.size
        self.fwhm = fwhm

        if self.psf is None:
            # (2*fwhm) is considered to avoid the boundary effect after the convolution
            self.size = np.array([2 * fwhm, 2 * fwhm])
            ideal_size = 2 ** np.ceil(np.log(self.im_size + 2 * self.size) / math.log(2))
            # to prevent any problems in the future, self.size has been forced to be an array of int
            self.size = np.array((ideal_size - self.im_size) / 2, dtype=int)

            y, x = np.indices((self.im_size[0] + 2 * self.size[0], self.im_size[1] + 2 * self.size[1]))
            sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
            self.psf = (1. / (2 * math.pi * sigma ** 2)) * np.exp(-((x - self.im_size[1] / 2 - self.size[1]) ** 2 + (y - self.im_size[0] / 2 - self.size[
                        0]) ** 2) / (2.0 * sigma ** 2))

            # normalization in order to ensure flux conservation
            self.psf /= self.psf.sum()
        else:
            self.size = np.array(img_psf.shape)
            self.psf = np.zeros((self.im_size[0] + 2 * self.size[0], self.im_size[1] + 2 * self.size[1]))
            self.psf[self.im_size[0]/2+self.size[0]/2:self.im_size[0]/2+3*self.size[0]/2, self.im_size[1]/2+self.size[1]/2:self.im_size[1]/2+3*self.size[1]/2]\
                = img_psf
            self.psf /= self.psf.sum()

        self.psf_fft2 = fft2(fftshift(self.psf))

    def convolution(self, data):
        """

        :param ndarray data:
        :return ndarray:
        """

        data2 = np.zeros((data.shape[0] + 2 * self.size[0], data.shape[1] + 2 * self.size[1]))
        data2[self.size[0]:data.shape[0] + self.size[0], self.size[1]:data.shape[1] + self.size[1]] = data

        data_conv = ifft2(fft2(data2) * self.psf_fft2)

        return data_conv[self.size[0]:data.shape[0] + self.size[0], self.size[1]:data.shape[1] + self.size[1]].real
