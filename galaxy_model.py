#!/usr/bin/env python

#### Pyhton library #####
import math
import numpy as np
from matplotlib import pyplot as plt

#### Program files ####
import tools
import velocity_model as VM


#######################   MODEL OF GALAXY  ########################

def exponential_disk_intensity(b0, rd, xcen, ycen, pa, incl, rtrunc, im_size=240, plot=False):
    """
    Function that computes the intensity map of an exponential disk

    Parameters
    ----------
    :b0: float
        central surface brightness (erg/s/arcsec2)
    rd: float
        disk scale length (arcsec)
    xcen: integer
        abscissa of the center in pixel
    ycen: integer
        ordinates of the center (arcsec)
    pa: float
        position angle of the major axis (degree)
    incl: float
        inclination of the disk (degree)
    rtrunc: integer
        truncation radius in pixel (after which the flux is null)
    im_size: integer
        maximum radius of the scene in pixel, im_size should be larger than the slit length + seeing (Default im_size=100)
    res: float
        resolution of the high resolution data (arcsec), res should be at least n x pixel size (Default res=0.04")
    plot: bool
        keyword to show a figure of the result (Default id False)

    """

    y, x = np.indices((im_size, im_size))  # index matrices
    den = (y - ycen) * math.cos(math.radians(pa)) - (x - xcen) * math.sin(math.radians(pa))
    num = - (x - xcen) * math.cos(math.radians(pa)) - (y - ycen) * math.sin(math.radians(pa))
    r = (den ** 2 + (num / math.cos(math.radians(incl))) ** 2) ** 0.5

    # Calculation of the surface brightness in the plane of the galaxy
    if rd != 0:
        b = b0 * np.exp(- np.abs(r) / rd)
    else:
        b = b0 * np.exp(0 * r)

    b[np.where(r > rtrunc)] = 0.

    # Display of the result
    if plot is True:
        plt.figure(101)
        plt.imshow(b, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.title('exponential_disk_intensity')
        plt.draw()

    return b


def disk_velocity(vel_model, v_max, rd, xcen, ycen, pa, incl, im_size=240, res=0.05, plot=False):
    """
    Function that computes the velocity field for the chosen model

    Parameters100
    ----------
    vel_model: function
        name of the function of the velocity profile
    v_max: float
        Maximum rotation velocity (km/s)
    rd: integer
        characteristic radius in pixel
    xcen: intger
        abscissa of the center (arcsec)
    ycen: intger
        ordiantes of the center (arcsec)
    pa: float
        position angle of the major axis (degree)
    incl: float
        inclination of the disk (degree)
    im_size: integer
        maximum radius of the scene (arcsec),
        im_size should be larger than the slit length + seeing (Default im_size=100)
    res: float
        resolution of the high resolution data (arcsec),
        res should be at least n x pixel size (Default res=0.04)
    plot: bool
        keyword to show a figure of the result (Default is F100alse)

    """

    # Conversion from Sky coordinates to Galatic coordinates
    r, ctheta = tools.sky_coord_to_galactic(xcen, ycen, pa, incl, im_size=(240, 240))

    # Calculation of the rotational velocity in the plane of the galaxy
    vr = vel_model(r, rd, v_max)

    # Calculation of the velocity field
    v = vr * math.sin(math.radians(incl)) * ctheta

    # Display of the result
    if plot is True:
        plt.figure(102)
        plt.imshow(v, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.title('disk_velocity')
        plt.draw()

    return v


def linear_velocity_dispersion(sig0, slope, xcen, ycen, pa, incl, im_size=240, res=0.05, plot=False):
    """
    Function that computes the velocity b[np.where(r > rtrunc)] = 0.dispersion map with a linearly decreasing profile

    Parameters
    ----------
    sig0: float
        velocity dispersion (km/s)
    slope: float
        slope of the velocity dispersion profile (km/s / arcsec)
    xcen: float
        abscissa of the center (arcsec)
    ycen: float
        ordinates of the center (arcsec)100
    pa: float
        position angle of the major axis (degree)
    incl: float
        inclination of the disk (deg552683ree)b[np.where(r > rtrunc)] = 0.
    im_size: float
        maximum radius of the scene (arcsec),
        im_size should be larger than the slit length + seeing (Default im_size=100)
    res: float
        resolution of the high resolution data (arcsec),
        res should be at least n x pixel size (Default res=0.04)
    plot: bool
        keyword to show a figure of the result (Default is False)

    """

    # Conversion from Sky coordinates to Galactic coordinates
    r, ctheta = tools.sky_coord_to_galactic(xcen, ycen, pa, incl, im_size=(240, 240))

    # Calculation of the velocity dispersion
    sig = sig0 + slope * np.abs(r)
    sig[np.where(sig <= 0)] = 0

    # Display of the result
    if plot is True:
        plt.figure(103)
        plt.imshow(sig, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.title('linear_velocity_dispersion')
        plt.draw()

    return sig


def velocity_map(v_max, rd, xcen, ycen, pa, incl, fwhm, b0, rtrunc, pix, im_size=240., res=0.05, plot=False,
                 args=(exponential_disk_intensity, VM.exponential_velocity)):
    """
    Compute the velocity  map in high definition. Rebin has been done after a convolution by a gaussian psf.

     Parameters
    ------------
    v_max: float
        Maximum rotation velocity (km/s)
    rd: float
        characteristic radius (arcsec)
    xcen: integer
        abscissa of the center in pixel
    ycen: float
        ordinates of the center in pixel
    pa: float
        position angle of the major axis (degree)
    incl: float
        inclination of the disk (degree)
    sig0: float
        velocity dispersion (km/s)
    slope: float
        slope of the velocity dispersion profile (km/s / arcsec)    im_size
    fwhm: float
        full width at half maximum of the Gaussian function which represents the PSF (arcsec)
    pix: float
        size of a pixel using for observations (arcsec)
        needed to compute the new map's resolution
    b0: float
        central surface brightness (erg/s/arcsec2)
    rtrunc: integer
        truncation radius in pixel (after which the flux is null)
    im_size: integer
        maximum radius of the scene (arcsec),
        im_size should be larger than the slit length + seeing (Default im_size=100)
    res: float
        resolution of the high resolution data (arcsec),
        res should be at least n x pixel size (Default res=0.04)
    plot: bool
        keyword to show a figure of the result (Default is False)
    args: tuple
        array containing the name of the function of the intensity and velocity profil
        by default args=[exponential_disk_intensity, VM.exponential_velocity]
    """

    flux = args[0](b0, rd, xcen, ycen, pa, incl, rtrunc, im_size=im_size, plot=plot)
    flux_conv = tools.rebin_data(tools.psf_convolution(flux, img_psf=None, fwhm=fwhm, im_size=im_size), pix/res)

    vel_map = disk_velocity(args[1], v_max, rd, xcen, ycen, pa, incl, im_size, res=res, plot=plot)

    vel_times_flux = tools.rebin_data(tools.psf_convolution(flux * vel_map, img_psf=None, fwhm=fwhm, im_size=im_size), pix/res)

    vel_final = np.zeros(np.shape(vel_times_flux))

    truediv_ind = np.where(flux_conv/np.max(flux_conv) > 1e-4)

    vel_final[truediv_ind] = vel_times_flux[truediv_ind]/flux_conv[truediv_ind]

    return vel_final


def square_vel_disp(v_max, rd, xcen, ycen, pa, incl, sig0, slope, fwhm, b0, rtrunc, im_size=240, res=0.05,
                    args=(exponential_disk_intensity, VM.exponential_velocity)):
    """
    Compute the velocity dispersion map in high definition. Rebin has been done after a convolution by a gaussian psf.

    Parameters
    ------------
    v_max: float
        Maximum rotation velocity (km/s)
    rd: float
        characteristic radius (arcsec)
    xcen: float
        abscissa of the center (arcsec)
    ycen: float
        ordiantes of the center (arcsec)
    pa: float
        position angle of the major axis (degree)
    incl: floattools.rebin_data(vel_times_flux, pix/delta_xy
        inclination of the disk (degree)
    sig0: float
        velocity dispersion (km/s)
    slope: float
        slope of the velocity dispersion profile (km/s / arcsec)
    fwhm: float
        full width at half maximum of the Gaussian function which represents the PSF (arcsec)
    pix: float
        size of a pixel using for observations (arcsec)
        needed to compute the new map's resolution
    b0: flt
        central surface brightness (erg/s/arcsec2)
    im_size: float
        maximum radius of the scene (arcsec),
        im_size should be larger than the slit length + seeing (Default im_size=100)
    rtrunc : integer
        truncation radius in pixel (after which the flux is null)
    res: float
        resolution of the high resolution data (arcsec),
        res should be at least n x pixel size (Default res=0.04)
    args: tuple
        array containing the name of the function of the intensity and velocity profil
        by default args=[exponential_disk_intensity, VM.exponential_velocity]
    """

    flux = args[0](b0, rd, xcen, ycen, pa, incl, rtrunc, im_size=im_size)
    vel = disk_velocity(args[1], v_max, rd, xcen, ycen, pa, incl, im_size=im_size, res=res)
    sig = linear_velocity_dispersion(sig0, slope, xcen, ycen, pa, incl, im_size=im_size, res=res)

    flux_conv = tools.rebin_data(tools.psf_convolution(flux, img_psf=None, fwhm=fwhm, im_size=im_size), pix/res)

    term1 = tools.rebin_data(tools.psf_convolution(sig ** 2 * flux, img_psf=None, fwhm=fwhm, im_size=im_size), pix/res) / flux_conv
    term2 = tools.rebin_data(tools.psf_convolution(vel ** 2 * flux, img_psf=None, fwhm=fwhm, im_size=im_size), pix/res) / flux_conv
    term3 = (tools.rebin_data(tools.psf_convolution(vel * flux, img_psf=None, fwhm=fwhm, im_size=im_size), pix/res) / flux_conv) ** 2

    return term1 + term2 - term3


if __name__ == '__main__':
    # MUSE data resolution
    pix = 0.2  # arcsec (pixel scale) 0.2" WFM or 0.025 NFM
    dl_pix = 1.25  # A/pixel (grism dispersion)exponential_disk_intensity
    R = 2500.
    # High resolution of the model:
    im_size = 60  # size in pixel of th high resolution model (size*size)
    # center of th galaxy modeled
    xcen = 30
    ycen = 30
    rtrunc = 10  # truncation radius in pixel (after which the flux is null)
    rd = 5  # characteristic radius in pixel

    fwhm = 2.5  # FWHM of a gaussian which represent the PSF in pixel
    # seeing = 2    # seeing observation in arcsec (FWHM of a 2D gaussian)

    model_res = pix/4  # Spatial resolution of the model

    pa = 0.  # position angle in degree
    incl = 45.  # inclination in degree
    b0 = 1000.  # central surface brightness

    v_max = 100.  # km/s
    vs = 0.  # systemic velocity in km/s
    sig0 = 20  # velocity dispersion km/s

    slope = 0.  # slope of the velocity dispersion profile


    #  TEST  #
    # Observed relosution to model resolution
    rd, xcen, ycen, fwhm, rtrunc, im_size = tools.higher_resolution_model(rd, xcen, ycen, fwhm, rtrunc, im_size, pix, model_res)

    #  Flux, Gaussian convolution and rebin image tests
    # flux = exponential_disk_intensity(b0, rd, xcen, ycen, pa, incl, rtrunc, im_size=im_size, plot=False)
    # tools.create_fits_file(flux, 'test_flux', overwrite=True)
    #
    # flux_conv = tools.psf_convolution(flux, img_psf=None, fwhm=fwhm, im_size=im_size)
    # tools.create_fits_file(flux_conv,'test_flux_conv', overwrite=True)
    #
    # flux_conv_rebin = tools.rebin_data(flux_conv, pix/model_res)
    # tools.create_fits_file(flux_conv_rebin, 'test_flux_conv_rebin', overwrite=True)

    # Velocity test
    vel = velocity_map(v_max, rd, xcen, ycen, pa, incl, fwhm, b0, rtrunc, pix, im_size=im_size, res=model_res,
                       plot=False, args=(exponential_disk_intensity, VM.exponential_velocity))
    # tools.create_fits_file(vel, 'test_vel', overwrite=True)
    # plt.figure()
    # plt.imshow(vel, origin='lower', interpolation='nearest')
    # plt.show()

    # Dispersion test
    # disp = np.sqrt(square_vel_disp(v_max, rd, xcen, ycen, pa, incl, sig0, slope, fwhm, b0, rtrunc, im_size=im_size,
    #                                res=model_res, args=(exponential_disk_intensity, VM.exponential_velocity)))
    # tools.create_fits_file(disp, 'test_disp', overwrite=True)

    ##########################################################
    # plt.draw() doesn't block the execution of the program
    # But you need to use plt.show() at the end to display them
    # Show you all images drew at the same time
    # I set by default Intensity(101), Velocity(102) and Dispersion(103)
    # plt.show()
    ##########################################################