import math
import numpy as np
from numpy.fft import fftshift, fft2, ifft2
from Class.Images import Image
from scipy.interpolate import interp2d
import logging
from astropy.io import fits
from Tools import io_stream
import sys

logger = logging.getLogger('__galcin__')


class PSF:
    """
        Create a 2D gaussian as PSF, store all the information about the psf.

        Perform the convolution af an image with the psf stored by the method 'convolution'.
        You can initiate this class with a fits file of a PSF, in this case no gaussian will be created.

        To avoid (at least minimize) any border effect, class PSF need information
            about the size of the highest resolution image passed by a Image's class.
    """

    def __init__(self, flux, psf_file=None, path=None, fwhm_lr=None, smooth=None):
        """
            constructor of the PSF class

        :param Image flux: class which represent the flux
        :param str psf_file: name of the fits file of the psf
        :param str path: path where is the fits file of the psf
        :param float fwhm_lr: fwhm of a gaussian
        :param float smooth: size of the smooth used on image
        """

        if psf_file is None:
            self.flux_size = flux.size
            self.fwhm = fwhm_lr*flux.oversample
            self.smooth = smooth*flux.oversample
            self.fwhm_f = np.sqrt(self.fwhm**2 + self.smooth**2)
            # (2*fwhm) is considered to avoid the boundary effect after the convolution
            self.size = np.array([2 * self.fwhm_f, 2 * self.fwhm_f])
            ideal_size = 2 ** np.ceil(np.log(self.flux_size + 2 * self.size) / math.log(2))
            # to prevent any problems in the future, self.size has been forced to be an array of int
            self.size = np.array((ideal_size - self.flux_size) / 2, dtype=int)

            y, x = np.indices((self.flux_size[0] + 2 * self.size[0], self.flux_size[1] + 2 * self.size[1]))
            sigma = self.fwhm_f / (2 * math.sqrt(2 * math.log(2)))
            self.psf = (1. / (2 * math.pi * sigma ** 2)) * np.exp(-((x - self.flux_size[1] / 2 - self.size[1]) ** 2 + (y - self.flux_size[0] / 2 - self.size[
                0]) ** 2) / (2.0 * sigma ** 2))

            # normalization in order to ensure flux conservation
            self.psf /= self.psf.sum()
            logger.debug('2D gaussian with fwhm={} pixels in high resolution'.format(self.fwhm_f))

        elif psf_file is str:
            img_psf = fits.getdata(io_stream.search_file(path, psf_file))
            self.header = fits.getheader(io_stream.search_file(path, psf_file))

            logger.debug('import psf from {}'.format(io_stream.search_file(path, psf_file)))

            self.calc_pix_size()
            img_psf_interp = self.interpolation(img_psf, int(self.pix_size_h/flux.get_oversamp()))
            self.size = np.array(img_psf_interp.shape)

            self.psf = np.zeros((self.flux_size[0] + 2 * self.size[0], self.flux_size[1] + 2 * self.size[1]))
            self.psf[self.flux_size[0]/2+self.size[0]/2:self.flux_size[0]/2+3*self.size[0]/2,
                     self.flux_size[1]/2+self.size[1]/2:self.flux_size[1]/2+3*self.size[1]/2] = img_psf_interp
            self.psf /= self.psf.sum()

        else:
            logger.warning('input psf parameters does not match, see the help')
            sys.exit()

        self.psf_fft2 = fft2(fftshift(self.psf))

    def interpolation(self, img_psf, oversamp):
        """
            Perform an interpolation of the image at higher resolution and stock the new image

        :param ndarray img_psf:
        :param int oversamp:
        """

        y = np.linspace(0, img_psf.shape()[0], img_psf.shape()[0])
        x = np.linspace(0, img_psf.shape()[1], img_psf.shape()[1])
        new_y = np.linspace(0, img_psf.shape()[0], img_psf.shape()[0] * int(oversamp))
        new_x = np.linspace(0, img_psf.shape()[1], img_psf.shape()[1] * int(oversamp))

        data = np.reshape(img_psf, -1)

        func = interp2d(x, y, data, kind='linear', fill_value=0)

        self.pix_size_h *= oversamp
        self.pix_size_l *= oversamp

        return np.array(func(new_x, new_y)).transpose()

    def calc_pix_size(self):
        try:
            self.pix_size_h = np.sqrt(self.header['CD1_1']**2 + self.header['CD1_2'])
            self.pix_size_l = np.sqrt(self.header['CD2_1']**2 + self.header['CD2_2'])
        except KeyError:
            self.pix_size_h = self.header['CDELT1']
            self.pix_size_l = self.header['CDELT2']
        logger.debug('PSF : pixel size in rad: {} (high) and {} (length)'.format(self.pix_size_1, self.pix_size_2))

    def convolution(self, data):
        """
            Do the convolution product between the data and the PSF

        :param ndarray data:
        :return ndarray:
        """

        data2 = np.zeros((data.shape[0] + 2 * self.size[0], data.shape[1] + 2 * self.size[1]))
        data2[self.size[0]:data.shape[0] + self.size[0], self.size[1]:data.shape[1] + self.size[1]] = data

        data_conv = ifft2(fft2(data2) * self.psf_fft2)
        data_conv = data_conv[self.size[0]:data.shape[0] + self.size[0], self.size[1]:data.shape[1] + self.size[1]].real

        return data_conv
