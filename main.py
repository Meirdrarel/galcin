#!/usr/bin/env python

# Pyhton library
import sys
import os
import time
import numpy as np
from astropy.io import ascii

# Program files
from PSF import PSF
from Images import Image, ImageOverSamp
import velocity_model as vm
from use_mpfit import use_mpfit

def main(argv):
    # HANDLE OF PROGRAMS ARGUMENTS
    try:
        vel_model = argv[0]
        path = argv[1]
        input_param = argv[2]
        options = str(argv[3])
    except IndexError:
        print('Arguments error, use --help for more information')
        sys.exit()

    if argv[4]:
        slope = float(argv[4])
    else:
        slope = 0.

    # ADD NEW MODEL IN THIS DICTIONARY:
    model_name = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

    # OPEN FILE PARAMETER
    try:
        params = ascii.read(path)
    except FileNotFoundError:
        print('File not found in {}'.format(path))
        sys.exit()

    print('\nparameter file found')

    flux_ld = Image()
    flux_hd = ImageOverSamp(, params[7])
    vel = Image()
    errvel = Image()
    img_psf = None
    print('images imported successfully')

    model_name = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

    psf = PSF(flux_hd, img_psf, fwhm=np.sqrt(params[8]**2+params[10]**2))

    use_mpfit(psf, flux_ld, flux_hd, vel, errvel, params, model_name, slope=0)

if __name__ == '__main__':
    main(sys.argv[1:])