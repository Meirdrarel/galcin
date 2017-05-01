from mpi4py import MPI
import argparse
import sys
import numpy as np
from astropy.io import ascii, fits
from Class.PSF import PSF
from Class.Images import Image, ImageOverSamp
import Tools.velocity_model as vm
from use_mpfit import use_mpfit
from use_pymultinest import use_pymultinest
import Tools.tools
import math
from numpy.fft import fftshift, fft2, ifft2, irfft, rfft



class Model3D:
    def __init__(self):

        self.im_size = None
        self.pos_angl = None
        self.incl = None
        self.xcen = None
        self.ycen = None
        self.vmax = None
        self.syst_vel = None
        self.sig0 = 20
        self.slope = None
        self.rd = None
        self.radius = None
        self.theta = None
        self.vel_map = None
        self.center_bright = 2000
        self.fwhm = None
        self.smooth = None
        self.psf = None
        self.psf_size = None

    def disk_velocity(self, vel_model):
        """

        :param vel_model:
        :return ndarray:
        """

        vr = vel_model(self.radius, self.rd, self.vmax)

        # Calculation of the velocity field
        v = vr * math.sin(math.radians(self.incl)) * self.theta + self.syst_vel

        return v

    def exponential_disk_intensity(self):

        r, theta = tools.sky_coord_to_galactic(self.xcen, self.ycen, self.pos_angl, self.incl, im_size=self.im_size)

        if self.rd != 0:
            flux = self.center_bright * np.exp(- np.abs(r) / self.rd)
        else:
            flux = self.center_bright * np.exp(0 * r)

        # flux[np.where(r > self.rtrunc)] = 0.

        return flux

    def spectral_convolution(self, cube, kernel, lbda, lrange=16.8, lres=0.15):

        return
