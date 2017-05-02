#! usr/bin/env python
import numpy as np
import Tools.tools as tools
import matplotlib.pyplot as plt
import Tools.velocity_model as vm
from Class.Model2D import Model2D
from Class.PSF import PSF
from Class.Images import ImageOverSamp, Image


def create_vel_model(pos_angl, incl, xcen, ycen, vmax, syst_vel, sig0, charac_rad, fwhm, plot=False):
    """

    :param float pos_angl: 
    :param float incl: 
    :param float xcen: 
    :param float ycen: 
    :param float vmax: 
    :param float syst_vel: 
    :param float sig0: 
    :param float charac_rad: 
    :param float fwhm: 
    :param Union[ndarray,Iterable] size: 
    :param bool plot: 
    :return: 
    """

    flux_ld = Image('../validation/flux_ldImp.fits')
    # flux_hd = ImageOverSamp('flux_ld.fits', charac_rad)
    flux_hd = Image('../validation/flux_hdImp.fits')

    img_psf = None
    psf = PSF(flux_hd, img_psf, fwhm=fwhm)
    model = Model2D(flux_ld, flux_hd, sig0, slope=slope)
    model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_hd)
    model.velocity_map(psf, flux_ld, flux_hd, vm.exponential_velocity)

    tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, sig0, model.vel_map, '../validation/velImp')

    if plot is True:
        plt.figure()
        plt.imshow(model.vel_map, cmap='nipy_spectral')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':

    # Correspond to a final image size : 60*60 pixels
    pos_angl = 0
    incl = 45.
    xcen = 30
    ycen = 30
    center_bright = 1e3
    vmax = 100.
    syst_vel = 0
    sig0 = 20.
    rtrunc = 20
    charac_rad = 5
    slope = 0.
    oversample = 1
    size = np.array([59, 61])
    # faire la somme quadratique des deux
    fwhm = 4
    smooth = 2

    create_vel_model(pos_angl, incl, xcen, ycen, vmax, syst_vel, sig0, charac_rad, np.sqrt(fwhm ** 2 + smooth ** 2), plot=False)
