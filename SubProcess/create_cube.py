#!usr/bin/env python
from Class.Model3D import Model3D
import Tools.flux_model as fm
import Tools.velocity_model as vm
from astropy.io import fits
import Tools.tools as tools
import numpy as np


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
vmax = 150   # km/s
pa = 0       # degree
incl = 45    # degree
vs = 0       # km/s
xcen = 50    # pixels
ycen = 50    # pixels
sig0 = 20    # km/s
lrange = 60  # angstrom


# vel_model = vm.flat_velocity
# flux_model = fm.flat_disk_intensity
vel_model = vm.flat_velocity
flux_model = fm.flat_disk_intensity


# for a field of 4"x4" => 20x20 pixels in muse and 100x100 pixels with hst

model = Model3D(xcen, ycen, pa, incl, vs, vmax, rd, sig0, flux_model, lbda0, deltal_ld, lrange, pix_size_hd, im_size=(100, 100), slope=0)

model.create_cube(vel_model, fwhm_psf_ld/pix_size_ld)
model.add_clump(10)
cube_conv = model.conv_psf(model.cube, fwhm_psf_ld/pix_size_ld)
cube_conv_SP = model.conv_lsf(cube_conv, fwhm_lsf_ld/deltal_ld)
model.write_fits(cube_conv_SP, '../../test/test_cube_hd_muse')

cube_rebin = tools.rebin_data(model.cube, 5)
model.write_fits(cube_rebin, '../../test/test_cube_rebin', oversample=5)

# alias qfitsview='QFitsView_3.1.linux64'


