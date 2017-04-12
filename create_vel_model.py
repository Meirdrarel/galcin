#! usr/bin/env python
import numpy as np
import tools
import matplotlib.pyplot as plt
import velocity_model as vm
from model_2D import Model2D
from PSF import PSF
from Images import ImageOverSamp, Image


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

    flux_ld = Image('validation/flux2.fits')
    flux_hd = ImageOverSamp('validation/flux2.fits', charac_rad)

    img_psf = None
    psf = PSF(flux_hd, img_psf, fwhm=np.sqrt(fwhm ** 2 + smooth ** 2))
    model = Model2D(flux_ld, flux_hd, sig0, slope=slope)
    model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_hd)
    model.velocity_map(psf, flux_ld, flux_hd, vm.exponential_velocity)

    tools.write_fits(xcen, ycen, pos_angl, incl, incl, syst_vel, vmax, charac_rad, sig0, model.vel_map, 'validation/modelV_python')

    if plot is True:
        plt.figure()
        plt.imshow(model.vel_map, cmap='nipy_spectral')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':

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
    size = np.array([120, 120])
    # faire la somme quadratique des deux
    fwhm = 4
    smooth = 2

    create_vel_model(pos_angl, incl, xcen, ycen, vmax, syst_vel, sig0, charac_rad, fwhm, size, plot=False)
