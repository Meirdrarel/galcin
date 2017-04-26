#!/usr/bin/env python
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import threading
import argparse
import sys
import numpy as np
from astropy.io import ascii, fits
from PSF import PSF
from Images import Image, ImageOverSamp
import velocity_model as vm
from use_mpfit import use_mpfit
from use_pymultinest import use_pymultinest

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()

parser.add_argument('model')
parser.add_argument('input_txt')
parser.add_argument('fits_ld')
parser.add_argument('fits_vel')
parser.add_argument('fits_evel')
parser.add_argument('-hd', '--high', dest="fits_hd")
parser.add_argument('-v', '--verbose', default=False, action='store_true', dest='verbose')
parser.add_argument('-slope', dest="slope", type=float, default=0.)
parser.add_argument('-psf', dest='psf', default=None)
group = parser.add_mutually_exclusive_group()
group.add_argument('--mpfit', action='store_false', default=False, dest='mpfit_multinest')
group.add_argument('--multinest', action='store_true', dest='mpfit_multinest')
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
except FileNotFoundError as e:
        print("File not found \n'%s'", e)
        sys.exit()

if args.psf:
    img_psf = fits.getdata(args.psf)
else:
    img_psf = None

if rank == 0:
    print(' all images found \n import the chosen module \n using {}x{} images'.format(vel.size[0], vel.size[1]))
    sys.stdout.flush()

model_name = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

psf = PSF(flux_hd, img_psf, fwhm=np.sqrt(params[9]**2+params[11]**2))
comm.barrier()

if args.mpfit_multinest is True:
    use_pymultinest(psf, flux_ld, flux_hd, vel, errvel, params, model_name[args.model], slope=args.slope, rank=rank, quiet=args.verbose)
else:
    if rank == 0:
        if args.verbose:
            quiet = 0
        else:
            quiet = 1
        use_mpfit(psf, flux_ld, flux_hd, vel, errvel, params, model_name[args.model], slope=args.slope, quiet=quiet)