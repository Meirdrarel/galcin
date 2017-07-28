import math
import numpy as np
import sys
from Class.Images import Image
import logging

logger = logging.getLogger('__galcin__')


def sky_coord_to_galactic(xc, yc, pa, incl, im_size=None):
    """
        Convert position from Sky coordinates to Galactic coordinates

    :param float xc: position of the center in arcsec
    :param float yc: position of the center in arcsec
    :param float pa: position angle of the major axis degree
    :param float incl: inclination of the disk in degree
    :param ndarray im_size: maximum radius of the scene (arcsec),
                          im_size should be larger than the slit length + seeing (Default im_size=100)
    :return list[ndarray, ndarray]: [r, theta]
    """
    y, x = np.indices(im_size)
    den = (y - yc) * math.cos(math.radians(pa)) - (x - xc) * math.sin(math.radians(pa))
    num = - (x - xc) * math.cos(math.radians(pa)) - (y - yc) * math.sin(math.radians(pa))
    r = (den ** 2 + (num / math.cos(math.radians(incl))) ** 2) ** 0.5
    tpsi = num * 1.

    tpsi[np.where(den != 0)] /= den[np.where(den != 0)]  # to avoid a NaN at the center
    den2 = math.cos(math.radians(incl)) ** 2 + tpsi ** 2
    sg = np.sign(den)  # sign
    ctheta = sg * (math.cos(math.radians(incl)) ** 2 / den2) ** 0.5  # azimuth in galaxy plane

    return [r, ctheta]


def rebin_data(data, factor):
    """
        Rebin an image.

    example: For rebin an image from 240x240 to 60x60 pixels, factor=5

    :param ndarray data: array to rebin
    :param int factor: rebin factor
    """
    if data.ndim == 2:
        data2 = data.reshape(int(data.shape[0] / factor), factor, int(data.shape[1] / factor), factor)
        return data2.mean(1).mean(2)

    if data.ndim == 3:
        data2 = data.reshape(data.shape[0], int(data.shape[1] / factor), factor, int(data.shape[2] / factor), factor)
        return data2.mean(2).mean(3)


def compar_resolution(flux, vel, oversamp=None):
    """
        Compare the resolution between the flux and the velocity,
        perform an interpolation of the flux if needed

    :param Image flux:
    :param Image vel:
    :return str: suffix for filenames
    """
    rap_size = flux.get_pix_size()/vel.get_pix_size()
    logger.debug('rapport of size is {}'.format(rap_size))
    if rap_size[0] == rap_size[1]:
        logger.debug('images are square')
        if np.array_equal(rap_size, [1, 1]):
            logger.debug('images have the same resolution')
            flux.set_oversamp(oversamp)
            flux.interpolation()
            whd = ''
        elif float(rap_size[0]).is_integer() and float(rap_size[1]).is_integer():
            logger.debug('the factor is an integer')
            flux.set_oversamp(int(rap_size[0]))
            whd = '_whd'
        else:
            new_rap = int(np.ceil(rap_size[0]))
            new_size = vel.get_size()*new_rap
            flux.set_oversamp(new_rap)
            flux.interpolation(new_size=new_size)
            whd = '_whd'
    else:
        logger.warning('the case of non square images is not implemented')
        sys.exit()
    logger.info('the oversample is {}'.format(flux.get_oversamp()))

    return whd
