import time
import json
import sys
import numpy as np
import scipy.stats, scipy
import pymultinest
import matplotlib.pyplot as plt

import tools
from Model2D import Model2D
from PSF import PSF


def use_pymultinest(psf, flux_ld, flux_hd, vel, errvel, params, vel_model, slope=0, quiet=False):
    """

    :param PSF psf: 
    :param Image flux_ld: 
    :param Image flux_hd: 
    :param Image vel: 
    :param Image errvel: 
    :param Union[ndarray, Iterable] params: 
    :param model_name: 
    :param float slope: 
    :param int quiet: 
    :return: 
    """

    gal, xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, sig0, fwhm, psfz, smooth = params

    model = Model2D(flux_ld, flux_hd, sig0, slope=slope)
    # model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_hd)
    # model.velocity_map(psf, flux_ld, flux_hd, vel_model)


    def prior(cube, ndim, nparams):
        cube[0] = 10 ** (cube[0]*4 - 2)            # no prior on xcen
        cube[1] = 10 ** (cube[1]*4 - 2)            # no prior on ycen
        cube[2] = 10 ** (cube[2] * 2.556)          # pos angl
        cube[3] = 10 ** (cube[3] * 1.903 + 0.698)  # incl
        cube[4] = 10 ** (cube[4] * 3)              # sys vel
        cube[5] = 10 ** (cube[5] * 3)              # vmax
        cube[6] = 10 ** (cube[6] * 1.176)          # charac rad

    def loglike(cube, ndim, nparams):

        model.set_parameters(cube[0], cube[1], cube[2]-180, cube[3], cube[4], cube[5], cube[6], flux_hd)
        model.velocity_map(psf, flux_ld, flux_hd, vel_model)
        step1 = (vel.data[flux_ld.mask] - model.vel_map[flux_ld.mask])**2/(2*errvel.data[flux_ld.mask]**2)
        log = np.log(np.sqrt(2*np.pi)*errvel.data[flux_ld.mask])

        return np.mean(step1*log)

    params = ['xcen', 'ycen', 'pa', 'incl', 'vs', 'vm', 'rd']
    n_params = len(params)

    pymultinest.run(loglike, prior, n_params, outputfiles_basename='test_', resume=False, verbose=quiet, max_iter=2000)
    json.dump(params, open('params.json', 'w'))

    output = pymultinest.Analyzer(n_params=n_params, outputfiles_basename='test_')
    b = output.get_mode_stats()

    print(b)

