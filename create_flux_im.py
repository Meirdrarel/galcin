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

# im_psf = np.zeros((60, 60))
# im_psf[30, 30] = 1
# # im_psf[0, 59] = 1
# # im_psf[59, 0] = 1
# # im_psf[59, 59] = 1

# Test expon1ential disc brightness
flux = tools.exponential_disk_intensity(xcen, ycen, pos_angl, incl, charac_rad, center_bright, rtrunc, im_size)
write_fits(xcen, ycen, pos_angl, incl, charac_rad, center_bright, rtrunc, oversample, syst_vel, sig0, vmax, flux, 'validation/flux2')

fluxC = Image('validation/flux2.fits')
psf = PSF(fluxC, img_psf=None, fwhm=np.sqrt(fwhm**2+smooth**2))

flux_conv = psf.convolution(fluxC.data)
flux_conv[np.logical_not(fluxC.mask)] = float('nan')

write_fits(xcen, ycen, pos_angl, incl, charac_rad, center_bright, rtrunc, oversample, syst_vel, sig0, vmax, flux_conv, 'validation/flux_conv2')

# plt.figure(1)
# plt.imshow(fluxC.data, cmap='nipy_spectral')
# plt.colorbar()
# plt.figure(2)
# plt.imshow(flux_conv, cmap='nipy_spectral')
# plt.colorbar()
# plt.figure(3)
# plt.imshow(truc.data-truc_conv, cmap='nipy_spectral')
# plt.colorbar()
# plt.show()

