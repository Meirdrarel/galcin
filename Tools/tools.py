import math
import numpy as np
from astropy.io import fits
import os
import sys
import yaml


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


def write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, rd, sig0, data, filename, oversample=1, mask=None):

    if mask is not None:
        data[np.logical_not(mask)] = float('nan')

    hdu = fits.PrimaryHDU(data=data)
    hdu.header.append(('PA', pos_angl, 'position angle in degree'))
    hdu.header.append(('INCL', incl, 'inclination in degree'))
    hdu.header.append(('XCEN', xcen / oversample, 'center abscissa in pixel'))
    hdu.header.append(('YCEN', ycen / oversample, 'center ordinate in pixel'))
    hdu.header.append(('RD', rd / oversample, 'characteristic radius in pixel'))
    hdu.header.append(('MAX_VEL', vmax, 'maximum velocity in km/s'))
    hdu.header.append(('SYST_VEL', syst_vel, 'systemic velocity in km/s'))
    hdu.header.append(('SIG0', sig0, 'dispersion velocity in km/s'))

    hdulist = fits.HDUList(hdu)
    hdulist.writeto(filename + '.fits', checksum=True, overwrite=True)


def search_file(path, filename):
    try:
        file_list = os.listdir(path)
        while True:
            if filename in file_list:
                if path != '.':
                    return path+filename
                else:
                    return filename
            else:
                print('File {} not found in directory {}'.format(filename, path))
                sys.exit()
    except FileNotFoundError:
        print('No such file or directory in {}'.format(path))
        sys.exit()


def make_dir(path, config):

    if path == '.':
        path = ''

    dirname = config['config fit']['method'] + '_' + config['config fit']['model']
    if config['config fit']['incfix'] or config['config fit']['xfix'] or config['config fit']['yfix']:
        dirname += '_'
    if config['config fit']['xfix']:
        dirname += 'x'
    if config['config fit']['yfix']:
        dirname += 'y'
    if config['config fit']['incfix']:
        dirname += 'i'
    if os.path.isdir(path+dirname) is False:
        print("\ncreate directory {}".format(dirname))
        os.makedirs(path+dirname)
    return path+dirname


def write_yaml(path, params, galname):
    outstream = open(path+'/results.yaml', 'w')

    paramstowrite = {'gal name': galname, 'results': ''}

    truc = {}
    for key in params['results']:
        truc.update({key: {'value': float(params['results'][key]['value']), 'error': float(params['results'][key]['error'])}})

    paramstowrite['results'] = truc

    try:
        paramstowrite.update({'mpfit stats': {'chi2r': float(params['mpfit']['chi2r']), 'dof': float(params['mpfit']['dof'])}})
    except KeyError:
        pass

    try:
        paramstowrite.update({'PymultiNest': {'log likelihood': params['PyMultiNest']['log likelihood']}})
    except KeyError:
        pass

    yaml.dump(paramstowrite, outstream, default_flow_style=False)
    outstream.close()
