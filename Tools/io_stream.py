import math
import numpy as np
from astropy.io import fits
import os
import sys
import yaml
import logging

logger = logging.getLogger('__galcin__')


def write_fits(data, filename, config, results, mask=None):
    """
        write data in fits file with model's parameters

    :param ndarray data: data to write in fits file
    :param str filename: name of the fits file (with path)
    :param dict config: config file
    :param dict results: dictionary of the results
    :param ndarray[bool] mask: boolean mask
    :return:
    """
    if mask is not None:
        data[np.logical_not(mask)] = float('nan')

    hdu = fits.PrimaryHDU(data=data)
    for key in config['init fit']['parname']:
        try:
            hdu.header.append((key, results['results'][key]['value'], config['init fit'][key]['desc']))
        except KeyError as k:
            hdu.header.append((key, results['results'][key]['value']))
            logger.exception("key 'desc' not found, parameter '{}' written without description".format(key), exc_info=k)

    hdulist = fits.HDUList(hdu)
    hdulist.writeto(filename + '.fits', checksum=True, overwrite=True)


def search_file(path, filename):
    """
        Search a file in a directory and all its subdirectories. Return the path of the file

    :param str path: path where to search
    :param str filename: name of the file to search (with its extension)
    :return str: the relative path where the file is
    """
    try:
        for root, dirs, files in os.walk(path):
            if filename in files:
                if root == '':
                    logger.debug("path is '' :"+filename)
                    return filename
                elif root == path and root is not '.':
                    toreturn = root+filename
                    logger.debug(toreturn)
                    return toreturn
                else:
                    toreturn = root+'/'+filename
                    logger.debug(toreturn)
                    return toreturn
        logger.error("File {} not found in directory {} and its subdirectories".format(filename, path))
        sys.exit()
    except FileNotFoundError as F:
        logger.exception('No such file or directory {}'.format(path), exc_info=F)
        sys.exit()


def make_dir(path, config):
    """
        Create the directory where results will be written
        The name of the directory depend of fixed parameters

    :param path: path where fits files are
    :param dict config: YAML config dictionary
    :return str: the path
    """

    if path == '.':
        path = ''

    dirname = config['config fit']['method'] + '_' + config['config fit']['model']

    if config['config fit']['method'] != 'multinest':
        suffix = ''
        for key in config['init fit']['parname']:
            if config['init fit'][key]['fixed'] == 1:
                suffix += key[0]

        if suffix != '':
            dirname += '_'+suffix

    if os.path.isdir(path+dirname) is False:
        logger.info("\ncreate directory {}".format(dirname))
        os.makedirs(path+dirname)
    toreturn = path+dirname
    logger.debug('makedir: {}'.format(toreturn))
    return toreturn


def write_yaml(path, params, galname, whd):
    """

        Setup the stream and write the YAML file which contain the results

    :param str path: path where to write the YAML file
    :param dict params: dictionary of the bestfit parameters
    :param str galname: the name of the galaxy
    :param str whd: suffix when a high resolution map is used
    :return:
    """

    outstream = open(path+'/results'+whd+'.yaml', 'w')

    dictowrite = {'gal name': galname, 'results': ''}

    sub_dict = {}
    for key in params['results']:
        sub_dict.update({key: {'value': float(params['results'][key]['value']), 'error': float(params['results'][key]['error'])}})

    dictowrite['results'] = sub_dict

    try:
        dictowrite.update({'mpfit stats': {'chi2r': float(params['mpfit']['chi2r']), 'dof': float(params['mpfit']['dof'])}})
    except KeyError:
        logger.debug("keyError: Key 'mpfit' not found in the results' dictionary")
        pass
    try:
        dictowrite.update({'PymultiNest': {'log likelihood': params['PyMultiNest']['log likelihood']}})
    except KeyError:
        logger.debug("keyError: Key 'PyMultiNest' not found in the results' dictionary")
        pass

    yaml.dump(dictowrite, outstream, default_flow_style=False)
    outstream.close()
