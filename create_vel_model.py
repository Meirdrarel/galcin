#! usr/bin/env python
import numpy as np
import tools
from astropy.io import fits
import matplotlib.pyplot as plt
import velocity_model as vm
from model_2D import Model2D
from PSF import PSF
from Images import ImageOverSamp, Image

# Correspond to a final image size : 60*60 pixels
pos_angl = 0
incl = 45.
xcen = 60
ycen = 50
center_bright = 1e3
vmax = 100.
syst_vel = 40.
sig0 = 20.
rtrunc = 30
charac_rad = 10
slope = 0.
oversample = 1
im_size = np.array([120, 120])
# faire la somme quadratique des deux
fwhm = 4
smooth = 2


def write_fits(xcen, ycen, pos_angl, incl, rd, center_bright, rtrunc, oversample, syst_vel, sig0, vmax, data, filename):

    data[np.where(data == 0)] = float('nan')

    hdu = fits.PrimaryHDU(data=data)
    hdu.header.append(('PA', pos_angl, 'position angle in degree'))
    hdu.header.append(('INCL', incl, 'inclination in degree'))
    hdu.header.append(('XCEN', xcen / oversample, 'center abscissa in pixel'))
    hdu.header.append(('YCEN', ycen / oversample, 'center ordinate in pixel'))
    hdu.header.append(('RD', rd / oversample, 'characteristic radius in pixel'))
    hdu.header.append(('MAX_VEL', vmax, 'maximum velocity in km/s'))
    hdu.header.append(('SYST_VEL', syst_vel, 'systemic velocity in km/s'))
    hdu.header.append(('SIG0', sig0, 'dispersion velocity in km/s'))
    hdu.header.append(('RTRUNC', rtrunc, 'truncated radius'))
    hdu.header.append(('B0', center_bright, ''))
    hdulist = fits.HDUList(hdu)
    hdulist.writeto(filename + '.fits', checksum=True, overwrite=True)


# create velocity model

flux_ld = Image('validation/flux2.fits')
flux_hd = ImageOverSamp('validation/flux2.fits', charac_rad)

img_psf = None
psf = PSF(flux_hd, img_psf, fwhm=np.sqrt(fwhm ** 2 + smooth ** 2))
model = Model2D(flux_ld, flux_hd, sig0, slope=slope)
model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_hd)
model.velocity_map(psf, flux_ld, flux_hd, vm.exponential_velocity)

write_fits(xcen, ycen, pos_angl, incl, charac_rad, center_bright, rtrunc, oversample, syst_vel, sig0, vmax, model.vel_map, 'validation/modelV_python')

plt.figure()
plt.imshow(model.vel_map, cmap='nipy_spectral')
plt.colorbar()
plt.show()

