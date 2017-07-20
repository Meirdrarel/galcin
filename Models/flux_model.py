import numpy as np
import Tools.tools as tools


def flat_disk_intensity(xc, yc, pa, incl, rd, center_bright, rtrunc, im_size):
    """

    :param float xc: position of the center in abscissa
    :param float yc: position of the center in ordinate
    :param float pa:
    :param float incl:
    :param float rd:
    :param float center_bright:
    :param float rtrunc:
    :param ndarray im_size:
    """

    r, theta = tools.sky_coord_to_galactic(xc, yc, pa, incl, im_size=im_size)

    flux = np.zeros(np.shape(r))

    flux[np.where(r <= rtrunc)] = center_bright

    flux[np.where(r > rtrunc)] = 0.

    return flux


def exponential_disk_intensity(xc, yc, pa, incl, rd, center_bright, rtrunc, im_size):
    """

    :param float xc:
    :param float yc:
    :param float pa:
    :param float incl: 
    :param float rd: 
    :param float center_bright: 
    :param float rtrunc: 
    :param ndarray im_size:
    """
    r, theta = tools.sky_coord_to_galactic(xc, yc, pa, incl, im_size=im_size)

    if rd != 0:
        flux = center_bright * np.exp(- np.abs(r) / rd)
    else:
        flux = center_bright * np.exp(0 * r)

    flux[np.where(r > rtrunc)] = 0.

    return flux
