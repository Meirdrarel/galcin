#!/usr/bin/env python3
import argparse
import os
import sys
import logging
import numpy as np
import yaml
from astropy.io import fits

import Models.velocity_model as vm
from Tools import calculus
from Tools import io_stream
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

            https://github.com/Meirdrarel/galcin

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
    input_stream = open(io_stream.search_file(path, filename), 'r')
    config = yaml.safe_load(input_stream)
    input_stream.close()
    files = config['files']
    params0 = config['init fit']
    confmod = config['conf model']

    # Search and open images file
    vel = Image(io_stream.search_file(path, files['vel']))
    flux = Image(io_stream.search_file(path, files['flux']))
    errvel = Image(io_stream.search_file(path, files['errvel']))
    if files['disp'] is not None:
        disp = Image(io_stream.search_file(path, files['disp']))

    ###################################################################################################
    # Test the size of the flux image with de velocity field and perform an interpolation if is needed
    if config['config fit']['oversamp'] or config['config fit']['oversamp'] is None:
        oversamp = 4
    else:
        oversamp = config['config fit']['oversamp']
    whd = calculus.compar_resolution(flux, vel, oversamp=oversamp)
    ###################################################################################################

    # Check psf file existence and import it or create a gaussian
    psf = PSF(flux, psf_file=files['psf'], path=path, fwhm_lr=confmod['psfx'], smooth=confmod['smooth'])

    # Do the convolution and the rebinning of the high resolution flux
    flux.conv_inter_flux(psf)

    # Initialise the model's class
    model = Model2D(vel, errvel, flux, psf, vm.vel_list[config['config fit']['model']], confmod['sig'], confmod['slope'])

    out_path = io_stream.make_dir(path, config)

    # Choose the method from the config file
    if config['config fit']['method'] == 'mpfit' and rank == 0:
        results = use_mpfit(model, params0, config['config fit']['mpfit'], quiet=config['config fit']['verbose'])

    elif config['config fit']['method'] == 'multinest':
        results = use_pymultinest(model, params0, config['config fit']['PyMultiNest'], quiet=config['config fit']['verbose'],
                                  pltstats=config['config fit']['PyMultiNest']['plt stats'], rank=rank, path=out_path, whd=whd)
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

        io_stream.write_fits(model.vel_map, out_path+'/modv'+whd, config, results, mask=vel.mask)

        io_stream.write_fits(vel.data-model.vel_map, out_path+'/resv'+whd, config, results, mask=vel.mask)

        io_stream.write_fits(model.vel_map_hd, out_path+'/modv_hd'+whd, config, results)

        io_stream.write_fits(model.vel_disp, out_path+'/modd'+whd, config, results, mask=vel.mask)

        if files['disp'] is not None:
            io_stream.write_fits(disp.data - model.vel_disp, out_path+'/resd'+whd, config, results, mask=vel.mask)

        io_stream.write_yaml(out_path, results, config['gal name'], whd=whd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('-h', '--help', description=galcin.__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('path', help="path to the directory where YAML file is")
    parser.add_argument('filename', help="Name of the YAML file which contain all parameters")
    args = parser.parse_args()

    galcin(path=args.path, filename=args.filename, rank=rank)


