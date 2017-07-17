#!/usr/bin/env python3
import argparse
import os
import sys
from astropy.io import ascii, fits
import numpy as np
import yaml

import Tools.tools as tools
import Tools.velocity_model as vm
from Class.Images import Image
from Class.PSF import PSF
from Class.Model2D import Model2D
from SubProcess.use_mpfit import use_mpfit
from SubProcess.use_pymultinest import use_pymultinest

try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 0:
        rank = MPI.COMM_WORLD.Get_rank()
except ImportError:
    rank = 0
    pass


def main(path=None, filename=None, rank=0):

    if rank == 0:
        try:
            print('\n entering in directory: {}'.format(path.split('/')[-2]))
        except IndexError:
            print('\n entering in directory: {}'.format(os.getcwd()))
    sys.stdout.flush()

    input_stream = open(tools.search_file(path, filename), 'r')
    config = yaml.safe_load(input_stream)
    input_stream.close()
    files = config['files']
    params0 = config['init fit']
    confmod = config['conf model']

    vel = Image(tools.search_file(path, files['vel']))
    flux = Image(tools.search_file(path, files['flux']))
    errvel = Image(tools.search_file(path, files['errvel']))
    if files['disp'] is not None:
        disp = Image(tools.search_file(path, files['disp']))

    # Test the size of the flux image with de velocity field and perform an interpolation if is needed
    length_rap = flux.length / vel.get_lenght()
    high_rap = flux.high / vel.get_high()
    if length_rap == 1 or high_rap == 1:
        if length_rap == high_rap:
            flux.set_oversamp(int(np.ceil(8 / params0['rt']['value'])))
            flux.interpolation()
            whd = ''
    else:
        whd = '_whd'

    # Check psf file existence
    img_psf = None
    if files['psf'] is not None:
        img_psf = fits.getdata(tools.search_file(path, files['psf']))
    psf = PSF(flux, img_psf=img_psf, fwhm_lr=confmod['psfx'], smooth=confmod['smooth'])

    # Do the convolution and the rebinning of the high resolution flux
    flux.conv_inter_flux(psf)

    model = Model2D(vel, errvel, flux, psf, vm.list_model[config['config fit']['model']], confmod['sig'], confmod['slope'])

    out_path = tools.make_dir(path, config)

    # Choose the method from the config file
    if config['config fit']['method'] == 'mpfit' and rank == 0:
        if config['config fit']['verbose'] is True:
            print('yolo')
        results = use_mpfit(model, params0, quiet=config['config fit']['verbose'])

    elif config['config fit']['method'] == 'multinest':

        conf_pymulti = config['config fit']['PyMultiNest']
        results = use_pymultinest(model, params0, quiet=config['config fit']['verbose'], nbp=conf_pymulti['nbp'], pltstats=conf_pymulti['plt stats'],
                                  rank=rank, path=out_path, whd=whd)

    else:
        print("Wrong fit's method input in the YAML config file"
              "\n use 'mpfit' for MPFIT or 'multinest' for PyMultiNest")
        sys.exit(1)

    if rank == 0:
        # Generation of maps with best fit parameters
        best_param = [results['results'][key]['value'] for key in params0['parname']]
        model.set_parameters(*best_param)
        model.velocity_map()
        model.vel_disp_map()

        tools.write_fits(*best_param, confmod['sig'], model.vel_map, out_path+'/modv'+whd, mask=vel.mask)

        tools.write_fits(*best_param, confmod['sig'], vel.data-model.vel_map, out_path+'/resv'+whd, mask=vel.mask)

        tools.write_fits(*best_param, confmod['sig'], model.vel_map_hd, out_path+'/modv_hd'+whd, oversample=1/flux.oversample)

        tools.write_fits(*best_param, confmod['sig'], model.vel_disp, out_path+'/modd'+whd, mask=vel.mask)

        if files['disp'] is not None:
            tools.write_fits(*results['params'], confmod['sig'], disp.data - model.vel_disp, out_path+'/resd'+whd, mask=vel.mask)

        tools.write_yaml(out_path, results, config['gal name'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="\t(name not found) fit model to velocity field of galaxies. "
                                                 "\n\tFor more information see the help or refer to the git repository:"
                                                 "\n\thttps://github.com/Meirdrarel/batman"
                                                 "\n\tdeveloped on python 3.6",
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('path', help="path to the directory where YAML file is")
    parser.add_argument('filename', help="Name of the YAML file which contain all parameters")
    args = parser.parse_args()

    main(path=args.path, filename=args.filename, rank=rank)


