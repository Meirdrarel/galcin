#!usr/bin/env python
import Tools.flux_model as fm
import Tools.velocity_model as vm
import Tools.tools as tools
from Class.Model3D import Model3D
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('filename', help='filename + sub directory in "/home/meirdrarel/Projet_PyCharm/StageM2/test"', type=str)
parser.add_argument('fm', help='flux model', type=str)
parser.add_argument('vm', help='velocity model', type=str)
parser.add_argument('incl', help='inclination', type=int)
parser.add_argument('vmax', help='velocity of the model', type=int)
parser.add_argument('-clump', dest='nb_cl', help='number of clump', type=int)
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
rd = 10      # pixels
vmax = args.vmax   # km/s
pa = 0       # degree
incl = args.incl    # degree
vs = 0       # km/s
xcen = 50    # pixels
ycen = 50    # pixels
sig0 = 20    # km/s
lrange = 50  # angstrom

fm_list = {'exp': fm.exponential_disk_intensity, 'flat': fm.flat_disk_intensity}
vm_list = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

# for a field of 4"x4" => 20x20 pixels in muse and 100x100 pixels with hst

model = Model3D(xcen, ycen, pa, incl, vs, vmax, rd, sig0, fm_list[args.fm], lbda0, deltal_ld, lrange, pix_size_hd, im_size=(100, 100), slope=0)
if args.nb_cl:
    model.create_cube(vm_list[args.vm], add_clump=True, nb_cl=args.nb_cl)
    ifclump = '_wc'
else:
    model.create_cube(vm_list[args.vm])
    ifclump = ''
cube_conv_smooth = model.conv_psf(model.cube, fwhm_psf_hd/pix_size_hd)
cube_conv_SP = model.conv_lsf(cube_conv_smooth, fwhm_lsf_ld/deltal_ld)
model.write_fits(cube_conv_SP, '../test/'+args.filename+ifclump)

cube_conv = model.conv_psf(cube_conv_SP, fwhm_psf_ld/pix_size_ld)
cube_rebin = tools.rebin_data(cube_conv, 5)
model.write_fits(cube_rebin, '../test/'+args.filename+ifclump+'_rebin', oversample=5)
