import math
import numpy as np
from numpy.fft import fftshift, fft2, ifft2
from Class.Images import Image


class PSF:

    def __init__(self, flux_hd, img_psf=None, fwhm_ld=3.5, smooth=0):
        """
        Create a 2D gaussian as PSF, store all the information about the psf.

        Perform the convolution af an image with the psf stored by the method 'convolution'.
        You can initiate this class with a fits file of a PSF, in this case no gaussian will be created.

        To avoid (at least minimize) any border effect, class PSF need information about the size of the biggest image passed by a Image's class.

        :param Image flux_hd:
        :param ndarray img_psf: image of the PSF
        :param float fwhm: fwhm of a gaussian, default is 3.5 pixels
        """

        self.psf = img_psf
        self.im_size = flux_hd.size
        self.fwhm = fwhm_ld*flux_hd.oversample
        self.smooth = smooth*flux_hd.oversample
        self.fwhm_f = np.sqrt(self.fwhm**2 + self.smooth**2)

        if self.psf is None:
            # (2*fwhm) is considered to avoid the boundary effect after the convolution
            self.size = np.array([2 * self.fwhm_f, 2 * self.fwhm_f])
            ideal_size = 2 ** np.ceil(np.log(self.im_size + 2 * self.size) / math.log(2))
            # to prevent any problems in the future, self.size has been forced to be an array of int
            self.size = np.array((ideal_size - self.im_size) / 2, dtype=int)

            y, x = np.indices((self.im_size[0] + 2 * self.size[0], self.im_size[1] + 2 * self.size[1]))
            sigma = self.fwhm_f / (2 * math.sqrt(2 * math.log(2)))
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
        data_conv = data_conv[self.size[0]:data.shape[0] + self.size[0], self.size[1]:data.shape[1] + self.size[1]].real

        return data_conv
