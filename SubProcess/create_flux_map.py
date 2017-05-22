#!usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))
import numpy as np
from numpy.fft import fftshift, fft2, ifft2
import math
import Tools.flux_model as fm
import Tools.tools as tools
import argparse
from astropy.io import ascii

parser = argparse.ArgumentParser()
parser.add_argument('path', help='directory where will create the map', type=str)
parser.add_argument('-fm', default='flat', help='flux model', type=str)
args = parser.parse_args()

fm_list = {'exp': fm.exponential_disk_intensity, 'flat': fm.flat_disk_intensity}

im_size = (300, 300)
xcen = 150
ycen = 150
pos_angl = 0
incl = 45
syst_vel = 0
vmax = 150
charac_rad = 15
rtrunc = 100
sig0 = 20
center_bright = 2000
fwhm = 20
smooth = 0
oversample = 5


def conv_psf(data, fwhm):
    # (2*fwhm) is considered to avoid the boundary effect after the convolution
    size = np.array([2 * fwhm, 2 * fwhm])
    ideal_size = 2 ** np.ceil(np.log(im_size + 2 * size) / math.log(2))
    # to prevent any problems in the future, size has been forced to be an array of int
    size = np.array((ideal_size - im_size) / 2, dtype=int)

    y, x = np.indices((im_size[0] + 2 * size[0], im_size[1] + 2 * size[1]))
    sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
    psf = (1. / (2 * math.pi * sigma ** 2)) * np.exp(-((x - im_size[1] / 2 - size[1]) ** 2 + (y - im_size[0] / 2 - size[0]) ** 2) / (2.0 * sigma ** 2))

    # normalization in order to ensure flux conservation
    psf /= psf.sum()

    data2 = np.zeros((data.shape[0] + 2 * size[0], data.shape[1] + 2 * size[1]))
    data2[size[0]:data.shape[0] + size[0], size[1]:data.shape[1] + size[1]] = data

    psf_fft2 = fft2(fftshift(psf))

    data_conv = ifft2(fft2(data2) * psf_fft2)
    data_conv = data_conv[size[0]:data.shape[0] + size[0], size[1]:data.shape[1] + size[1]].real

    return data_conv

flux = fm_list[args.fm](xcen, ycen, pos_angl, incl, charac_rad, center_bright, rtrunc, im_size)
fluxw = np.copy(flux)
fluxw[np.where(flux == 0)] = float('NaN')
tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, 0, fluxw, args. path +'fluxHD_python')
flux_rebin = tools.rebin_data(conv_psf(flux, np.sqrt(fwhm**2+smooth**2)), oversample)
flux_rebin[np.where(flux_rebin < 1)] = float('NaN')
tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, 0, flux_rebin, args. path +'flux_python')


params = np.array([xcen/oversample, ycen/oversample, pos_angl, incl, syst_vel, vmax, charac_rad/oversample, sig0, fwhm/oversample, smooth, oversample,
                   rtrunc/oversample])
names = ['x', 'y', 'pa', 'incl', 'vs', 'vm', 'rd', 'sig0', 'fwhm', 'smooth', 'oversample', 'rtrunc']
ascii.write(params, args.path+'param_model.txt', names=names, delimiter='\t', overwrite=True)
