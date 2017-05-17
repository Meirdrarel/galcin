#!/usr/bin/env python
import argparse
import sys

import numpy as np
from astropy.io import ascii, fits
from mpi4py import MPI

import Tools.tools as tools
import Tools.velocity_model as vm
from Class.Images import Image, ImageOverSamp
from Class.PSF import PSF
from SubProcess.use_mpfit import use_mpfit
from SubProcess.use_pymultinest import use_pymultinest

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()

parser.add_argument('model')
parser.add_argument('path')
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

params_file = tools.search_file(args.path, args.input_txt)
params = ascii.read(params_file)[0]

flux_ld_file = tools.search_file(args.path, args.fits_ld)
flux_ld = Image(flux_ld_file)

if args.fits_hd:
    flux_hd_file = tools.search_file(args.path, args.fits_hd)
    flux_hd = Image(flux_hd_file)
    flux_hd.oversample = int(flux_hd.length / flux_ld.length)

    whd = '_whd'
else:
    flux_hd = ImageOverSamp(flux_ld_file, params[7])
    flux_hd.oversample = int(flux_hd.length / flux_ld.length)
    whd = ''

vel_file = tools.search_file(args.path, args.fits_vel)
vel = Image(vel_file)

evel_file = tools.search_file(args.path, args.fits_evel)
errvel = Image(evel_file)


if args.psf:
    psf_file = tools.search_file(args.path, args.psf)
    img_psf = fits.getdata(psf_file)
else:
    img_psf = None

if rank == 0:
    print(' all images found \n import the chosen module \n using {}x{} images'.format(vel.size[0], vel.size[1]))
    sys.stdout.flush()

model_name = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

psf = PSF(flux_hd, img_psf, fwhm_ld=params[9], smooth=params[11])
if rank == 0:
    print(' using fwh  {} pixels for HD with oversample {}'.format(psf.fwhm_f, flux_hd.oversample))
    sys.stdout.flush()
comm.barrier()

flux_hd.conv_inter_flux(psf)

if args.mpfit_multinest is True:
    use_pymultinest(psf, flux_ld, flux_hd, vel, errvel, params, model_name[args.model], args.path, slope=args.slope, rank=rank, quiet=args.verbose,
                    whd=whd)
else:
    if rank == 0:
        if args.verbose:
            quiet = 0
        else:
            quiet = 1
        use_mpfit(psf, flux_ld, flux_hd, vel, errvel, params, model_name[args.model], args.path, slope=args.slope, quiet=quiet, whd=whd)
