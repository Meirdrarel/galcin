#!usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))
import Tools.flux_model as fm
import Tools.velocity_model as vm
import Tools.tools as tools
from Class.Model3D import Model3D
from Class.Clumps import Clumps
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('path', help='directory where will create cube', type=str)
parser.add_argument('incl', help='inclination', type=int)
parser.add_argument('vmax', help='velocity of the model', type=int)
parser.add_argument('-fm', default='flat', help='flux model', type=str)
parser.add_argument('-vm', default='flat', help='velocity model', type=str)
parser.add_argument('-clump', nargs='+', dest='ifclump', help='create clump', type=int)
parser.add_argument('-nocube', action='store_false', dest='ifcube', help='create cube', default=True)
args = parser.parse_args()

# parameters for low resolution (typically MUSE)
pix_size_ld = 0.2  # arcsec
deltal_ld = 1.25   # angstrom
lbda0 = 6562.78    # angstrom
fwhm_psf_ld = 0.7  # arcsec ==> 0.7/0.2 = 3.5 pixels
fwhm_lsf_ld = 2.5  # angstrom  == 2.5>/1.25 = 2 elements

# parameters for high resolution (typically HST)
pix_size_hd = 0.04  # arcsec
deltal_hd = 1.25    # angstrom
fwhm_psf_hd = 0.08  # arcsec
fwhm_lsf_hd = 2.5   # angstrom

# parameters for model
rd = 2*5      # pixels
vmax = args.vmax   # km/s
pa = 0       # degree
incl = args.incl    # degree
vs = 0       # km/s
xcen = 10*5    # pixels
ycen = 10*5    # pixels
sig0 = 30    # km/s
lrange = 50  # angstrom
rtrunc = 25

fm_list = {'exp': fm.exponential_disk_intensity, 'flat': fm.flat_disk_intensity}
vm_list = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

if args.ifcube:
    print('\nCreate Cube')
    # for a field of 4"x4" => 20x20 pixels in muse and 100x100 pixels with hst
    model = Model3D(xcen, ycen, pa, incl, vs, vmax, rd, rtrunc, sig0, fm_list[args.fm], lbda0, deltal_ld, lrange, pix_size_hd, im_size=(20*5, 20*5), slope=0)
    model.create_cube(vm_list[args.vm])

    # cube_conv_smooth = model.conv_psf(model.cube, fwhm_psf_hd/pix_size_hd+2)   # 2 pixels
    cube_conv_SP = model.conv_lsf(model.cube, fwhm_lsf_ld/deltal_ld)

    tools.write_fits(xcen, ycen, pa, incl, vs, vmax, rd, sig0, np.sum(cube_conv_SP, axis=0), args.path+'CUBE_flux_hd', oversample=5)
    model.write_fits(cube_conv_SP, args.path+'CUBE')

    cube_conv = model.conv_psf(cube_conv_SP, fwhm_psf_ld/pix_size_ld*5)   # 3.5*5 = 17.5 HST's pixels
    cube_rebin = tools.rebin_data(cube_conv, int(pix_size_ld/pix_size_hd))
    model.write_fits(cube_rebin, args.path+'CUBE_rebin', oversample=int(pix_size_ld/pix_size_hd))

    tools.write_fits(xcen, ycen, pa, incl, vs, vmax, rd, sig0, model.v, args.path+'CUBE_vel_map_hd')

    modv_ld = tools.rebin_data(model.v, int(pix_size_ld/pix_size_hd))
    tools.write_fits(xcen/5, ycen/5, pa, incl, vs, vmax, rd/5, sig0, modv_ld, args.path+'CUBE_vel_map', oversample=int(pix_size_ld/pix_size_hd))

if args.ifclump:
    print('\nCreate Clumps')
    # create clump
    clump = Clumps(xcen, ycen, pa, incl, vs, vmax, rd, rtrunc, sig0, lbda0, deltal_ld, lrange, pix_size_hd, im_size=(20*5, 20*5), slope=0)
    clump.create_clumps(args.ifclump, vm_list[args.vm], fwhm_lsf_ld/deltal_ld)
    clump.write_fits(clump.cube, args.path + 'CLUMP')

    clump_conv = clump.conv_psf(clump.cube, fwhm_psf_ld/pix_size_ld*5)   # 3.5*5 = 17.5 HST's pixels
    clump_rebin = tools.rebin_data(clump_conv, int(pix_size_ld/pix_size_hd))
    clump.write_fits(clump_rebin, args.path+'CLUMP_rebin', oversample=int(pix_size_ld/pix_size_hd))


if args.ifcube and args.ifclump:
    print('\nAdd clumps to the cube')
    cube_conv_SP += clump.cube
    model.write_fits(cube_conv_SP, args.path+'CUBE_wc', verbose=False)

    cube_rebin += clump_rebin
    model.write_fits(cube_rebin, args.path + 'CUBE_wc_rebin', oversample=int(pix_size_ld / pix_size_hd), verbose=False)

    tools.write_fits(xcen, ycen, pa, incl, vs, vmax, rd, sig0, np.sum(cube_conv_SP, axis=0), args.path + 'CUBE_wc_flux_hd', oversample=5)

