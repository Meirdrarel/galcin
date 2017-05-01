#! usr/bin/env python
import os
import numpy as np
import tools
import matplotlib.pyplot as plt
from PSF import PSF
from Images import Image


def exponential_disk_intensity(xcen, ycen, pos_angl, incl, rd, center_bright, rtrunc, im_size):
    """
    
    :param float xcen: 
    :param float ycen: 
    :param float pos_angl: 
    :param float incl: 
    :param float rd: 
    :param float center_bright: 
    :param float rtrunc: 
    :param ndarray im_size: 
    :return: 
    """
    r, theta = tools.sky_coord_to_galactic(xcen, ycen, pos_angl, incl, im_size=im_size)

    if rd != 0:
        flux = center_bright * np.exp(- np.abs(r) / rd)
    else:
        flux = center_bright * np.exp(0 * r)

    flux[np.where(r > rtrunc)] = 0.

    return flux


def create_map_flux(pos_angl, incl, xcen, ycen, vmax, syst_vel, sig0, rtrunc, charac_rad, fwhm, size, plot=False):
    """
    
    :param float pos_angl: 
    :param float incl: 
    :param float xcen: 
    :param float ycen: 
    :param float vmax: 
    :param float syst_vel: 
    :param float sig0: 
    :param int rtrunc: 
    :param float charac_rad: 
    :param float fwhm: 
    :param Union[ndarray,Iterable] size: 
    :param bool plot: 
    :return: 
    """

    im_size = np.array(size)
    center_bright = 1e3

    flux = exponential_disk_intensity(xcen, ycen, pos_angl, incl, charac_rad, center_bright, rtrunc, im_size)
    tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, sig0, flux, '../validation/flux_hdImp', oversample=1, chi2r=None, dof=None,
                     mask=None)

    flux_tmp = Image('../validation/flux_hdImp.fits')
    psf = PSF(flux_tmp, img_psf=None, fwhm=np.sqrt(fwhm**2+smooth**2))

    flux_conv = psf.convolution(flux_tmp.data)
    flux_conv[np.logical_not(flux_tmp.mask)] = float('nan')

    tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, sig0, flux_conv, '../validation/flux_ldImp', oversample=1, chi2r=None, dof=None,
                     mask=None )
    # os.remove('/tmp/flux_tmp.fits')

    if plot is True:
        plt.figure()
        plt.imshow(flux_conv, cmap='nipy_spectral')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    pos_angl = 0
    incl = 45.
    xcen = 30
    ycen = 30
    vmax = 100.
    syst_vel = 0
    sig0 = 20.
    rtrunc = 20
    charac_rad = 5
    slope = 0.
    oversample = 1
    size = np.array([59, 61])

    # do a quadratic sum of both
    fwhm = 4
    smooth = 2

    create_map_flux(pos_angl, incl, xcen, ycen, vmax, syst_vel, sig0, rtrunc, charac_rad, np.sqrt(fwhm**2+smooth**2), size)
