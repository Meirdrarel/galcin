import numpy as np
import Tools.tools as tools
from Class.Model2D import Model2D
from mpi4py import MPI
import argparse
import sys
from astropy.io import ascii, fits
from Class.PSF import PSF
from Class.Images import Image, ImageOverSamp
import Tools.velocity_model as vm
from use_mpfit import use_mpfit
from use_pymultinest import use_pymultinest
from numpy.fft import fftshift, fft2, ifft2, irfft, rfft


class Model3D(Model2D):
    def __init__(self, flux_ld, flux_hd, sig0, slope=0, lbda=6562.78, dlbda=1, lsize=40):
        Model2D.__init__(self, flux_ld, flux_hd, sig0, slope=0)

        self.cube = None
        self.lbd0 = lbda
        self.dlbda = dlbda
        self.lsize = lsize

    def spectral_convolution(self, cube, kernel, lbda, lrange=16.8, lres=0.15):

        #   CREATION OF THE GAUSSIAN FUNCTION
        # kernelind = np.int(np.ceil(kernel / lres))      # number of spatial sampling elements in kernel
        # ls = 2 * kernelind  # (2*fwhmind) is considered to avoid the boundary effect after the convolution.

        ideal_size = 2 ** np.ceil(np.log(self.lsize + 2 * ls) / np.log(2))
        ls = (ideal_size - self.lsize) / 2

        lindex = (np.arange(self.lsize + 2 * ls) - (self.lsize + 2 * ls) / 2) * lres  # spectral sampling elements
        #    vind = (lindex / lbda) * cst.c *1e-3              #velocity sampling elements

        sigma = kernel / (2 * np.sqrt(2 * np.log(2)))
        psfgrism = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(lindex) ** 2 / (2.0 * sigma ** 2))
        psfgrism /= psfgrism.sum()  # normalization in order to ensure flux conservation
        psfgrism_sh = fftshift(psfgrism)
        #   SPECTRAL CONVOLUTION
        cube2 = np.zeros((cube.shape[0] + 2 * ls, cube.shape[1], cube.shape[2]))
        cube2[ls:cube.shape[0] + ls, :, :] = cube

        specconv = irfft(rfft(cube2, axis=0) * rfft(psfgrism_sh.reshape(psfgrism_sh.shape[0], 1, 1), axis=0), axis=0)

        specconv = specconv.real
        specconv = specconv[ls:cube.shape[0] + ls, :, :]

        return specconv