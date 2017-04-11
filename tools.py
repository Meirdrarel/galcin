import math
import numpy as np
from PSF import PSF


def sky_coord_to_galactic(xcen, ycen, pos_angl, incl, im_size=(240, 240)):
    """
    Convert position from Sky coordinates to Galactic coordinates

    :param float xcen: position of the center in arcsec
    :param float ycen: position of the center in arcsec
    :param float pos_angl: position angle of the major axis degree
    :param float incl: inclination of the disk in degree
    :param ndarray im_size: maximum radius of the scene (arcsec),
                          im_size should be larger than the slit length + seeing (Default im_size=100)
    :param float res: resolution of the high resolution data (arcsec),
                      res should be at least n x pixel size (Default res=0.04)
    :return ndarray: [r, theta]
    """
    y, x = np.indices(im_size)
    den = (y - ycen) * math.cos(math.radians(pos_angl)) - (x - xcen) * math.sin(math.radians(pos_angl))
    num = - (x - xcen) * math.cos(math.radians(pos_angl)) - (y - ycen) * math.sin(math.radians(pos_angl))
    r = (den ** 2 + (num / math.cos(math.radians(incl))) ** 2) ** 0.5
    tpsi = num * 1.

    tpsi[np.where(den != 0)] /= den[np.where(den != 0)]  # to avoid a NaN at the center
    den2 = math.cos(math.radians(incl)) ** 2 + tpsi ** 2
    sg = np.sign(den)  # signe
    ctheta = sg * (math.cos(math.radians(incl)) ** 2 / den2) ** 0.5  # azimuth in galaxy plane
    
    return [r, ctheta]


def rebin_data(data, new_bin):
    """
    Rebin an image.

    :param ndarray data: array to rebin
    :param int new_bin: size of the new bin
    """

    data2 = data.reshape(int(data.shape[0] / new_bin), new_bin, int(data.shape[1] / new_bin), new_bin)
    
    return np.average(data2, axis=(1, 3))


def exponential_disk_intensity(xcen, ycen, pos_angl, incl, rd, center_bright, rtrunc, im_size):
    """
    
    :param PSF psf:
    :param float center_bright:
    :param int rtrunc:
    :param int oversample:
    """
    r, theta = sky_coord_to_galactic(xcen, ycen, pos_angl, incl, im_size=im_size)

    if rd != 0:
        flux = center_bright * np.exp(- np.abs(r) / rd)
    else:
        flux = center_bright * np.exp(0 * r)

    flux[np.where(r > rtrunc)] = 0.

    return flux
