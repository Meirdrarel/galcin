#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import numpy as np
import yaml
from astropy.io import fits

import Models.velocity_model as vm
import Tools.tools as tools
from Class.Images import Image
from Class.Model2D import Model2D
from Class.PSF import PSF
from SubProcess.use_mpfit import use_mpfit
from SubProcess.use_pymultinest import use_pymultinest

# Test if mpi4py is installed and if is it used
try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 0:
        rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0
    pass

# For debug use logging.DEBUG instead of logging.INFO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('__galcin__')


def galcin(path=None, filename=None, rank=0):
    """
        (name not found) fit model to velocity field of galaxies.
        For more information see the help or refer to the git repository:

            https://github.com/Meirdrarel/batman

        developed on python 3.6

        This program can be lunch from console:

            galcin.py (path) (filename)

        or in a python shell/console by importing this file and call:

            galcin(path, filename, rank)

        For run with mpi type in the prompt:

            mpiexec -n (nbcore) galcin.py (path) (filename)

        Output are written in a directory where the YAML file is,
           the directory name is:  (method)_(model)_(paramfix)

    :param str path: path where the config file is
    :param str filename: name of the config file
    :param int rank: id of the thread when the program is run with MPI4PY
    """

    if rank == 0:
        try:
            logger.info(' entering in directory: {}'.format(path.split('/')[-2]))
        except IndexError as I:
            logger.info(' entering in directory: {}'.format(os.getcwd()))

    # Open the YAML file and preset some parts
    input_stream = open(tools.search_file(path, filename), 'r')
    config = yaml.safe_load(input_stream)
    input_stream.close()
    files = config['files']
    params0 = config['init fit']
    confmod = config['conf model']

    # Search and open images file
    vel = Image(tools.search_file(path, files['vel']))
    flux = Image(tools.search_file(path, files['flux']))
    errvel = Image(tools.search_file(path, files['errvel']))
    if files['disp'] is not None:
        disp = Image(tools.search_file(path, files['disp']))

    ###################################################################################################
    # Test the size of the flux image with de velocity field and perform an interpolation if is needed
    length_rap = flux.length / vel.get_lenght()
    high_rap = flux.high / vel.get_high()
    if rank == 0:
        logger.warning(" change the calculation of the rapport of oversampling"
                       "\n\t\t  pixel size is not in the fits' header")
        logger.debug(" oversampling determine from size of images")
        logger.debug(" length: {} \thigh: {}".format(length_rap, high_rap))
    if length_rap == 1 or high_rap == 1:
        if length_rap == high_rap:
            flux.set_oversamp(int(np.ceil(8 / params0['rt']['value'])))
            flux.interpolation()
            whd = ''
    else:
        flux.set_oversamp(int(length_rap))
        whd = '_whd'
    ####################################################################################################

    # Check psf file existence and import it or create a gaussian
    img_psf = None
    if files['psf'] is not None:
        img_psf = fits.getdata(tools.search_file(path, files['psf']))
        logger.debug('import psf from {}'.format(tools.search_file(path, files['psf'])))
    else:
        logger.debug('2D gaussian with fwhm={} pixels in high resolution'.format(np.sqrt(confmod['psfx']**2+confmod['smooth']**2)*vel.oversample))
    psf = PSF(flux, img_psf=img_psf, fwhm_lr=confmod['psfx'], smooth=confmod['smooth'])

    # Do the convolution and the rebinning of the high resolution flux
    flux.conv_inter_flux(psf)

    # Initialise the model's class
    model = Model2D(vel, errvel, flux, psf, vm.list_model[config['config fit']['model']], confmod['sig'], confmod['slope'])

    out_path = tools.make_dir(path, config)

    # Choose the method from the config file
    if config['config fit']['method'] == 'mpfit' and rank == 0:
        results = use_mpfit(model, params0, quiet=config['config fit']['verbose'])
    elif config['config fit']['method'] == 'multinest':

        conf_pymulti = config['config fit']['PyMultiNest']
        results = use_pymultinest(model, params0, quiet=config['config fit']['verbose'], nbp=conf_pymulti['nbp'], pltstats=conf_pymulti['plt stats'],
                                  rank=rank, path=out_path, whd=whd)
    else:
        if rank == 0:
            logger.error(" wrong fit's method input in the YAML config file"
                         " use 'mpfit' for MPFIT or 'multinest' for PyMultiNest")
        else:
            logger.warning(" thread {} not used, for mpfit mpi is not necessary".format(rank))
        sys.exit()

    # Write fits file from the bestfit set of parameters and results in a YAML file
    if rank == 0:
        # Generation of maps with best fit parameters
        best_param = [results['results'][key]['value'] for key in params0['parname']]
        model.set_parameters(*best_param)
        model.velocity_map()
        model.vel_disp_map()

        tools.write_fits(model.vel_map, out_path+'/modv'+whd, config, results, mask=vel.mask)

        tools.write_fits(vel.data-model.vel_map, out_path+'/resv'+whd, config, results, mask=vel.mask)

        tools.write_fits(model.vel_map_hd, out_path+'/modv_hd'+whd, config, results)

        tools.write_fits(model.vel_disp, out_path+'/modd'+whd, config, results, mask=vel.mask)

        if files['disp'] is not None:
            tools.write_fits(*results['params'], confmod['sig'], disp.data - model.vel_disp, out_path+'/resd'+whd, mask=vel.mask)

        tools.write_yaml(out_path, results, config['gal name'], whd=whd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="\t(name not found) fit model to velocity field of galaxies. "
                                                 "\n\tFor more information see the help or refer to the git repository:\n"
                                                 "\n\t\thttps://github.com/Meirdrarel/batman\n"
                                                 "\n\tdeveloped on python 3.6\n"
                                                 "\n\tThis program can be lunch from console: \n"
                                                 "\n\t\tgalcin.py (path) (filename)\n"
                                                 "\n\tor in a python shell/console by importing this file and call: \n"
                                                 "\n\t\tgalcin(path, filename, rank)\n"
                                                 "\n\tFor run with mpi type in the prompt: \n"
                                                 "\n\t\tmpiexec -n (nbcore) galcin.py (path) (filename) \n"
                                                 "\n\tOutput are written in a directory where the YAML file is,"
                                                 "\n\tthe directory name is:  (method)_(model)_(paramfix)",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('path', help="path to the directory where YAML file is")
    parser.add_argument('filename', help="Name of the YAML file which contain all parameters")
    parser.add_argument('-h')
    args = parser.parse_args()

    galcin(path=args.path, filename=args.filename, rank=rank)


