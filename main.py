#!/usr/bin/env python

# Python library
import sys
import argparse
import os
import time
import numpy as np
from astropy.io import ascii

# Program files
from PSF import PSF
from Images import Image, ImageOverSamp
import velocity_model as vm
from use_mpfit import use_mpfit


parser = argparse.ArgumentParser()

parser.add_argument('model')
parser.add_argument('input_txt')
parser.add_argument('fits_ld')
parser.add_argument('fits_vel')
parser.add_argument('fits_evel')
parser.add_argument('-hd', '--high', dest="fits_hd")
parser.add_argument('-v', '--verbose', default=False, dest='verbose')
# parser.add_argument('-slope', dest="slope", type=float, const=0)

args = parser.parse_args()

# ADD NEW MODEL IN THIS DICTIONARY:

try:
    params = ascii.read(args.input_txt)[0]
    flux_ld = Image(args.fits_ld)
    if args.fits_hd:
        flux_hd = Image(args.fits_hd)
    else:
        flux_hd = ImageOverSamp(args.fits_ld, params[7])
    vel = Image(args.fits_vel)
    errvel = Image(args.fits_evel)
    img_psf = None
except FileNotFoundError:
    print("File ".format(FileNotFoundError))
    sys.exit()

if args.verbose:
    quiet = 0
else:
    quiet = 1

model_name = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

psf = PSF(flux_hd, img_psf, fwhm=np.sqrt(params[8]**2+params[10]**2))

use_mpfit(psf, flux_ld, flux_hd, vel, errvel, params, model_name[args.model], slope=0, quiet=quiet)
