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


def use_pymultinest(psf, flux_ld, flux_hd, vel, errvel, params, vel_model, slope=0, quiet=1):
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

    print((params)[0])
    # model = Model2D(flux_ld, flux_hd, sig0, slope=slope)
    # model.set_parameters(*params[1:8], flux_hd)
    # model.velocity_map(psf, flux_ld, flux_hd, vel_model)
    #
    # def prior(cube, ndim, nparams):
    #     # cube[0] = 10          # no prior on xcen
    #     # cube[1] = 10          # no prior on ycen
    #     cube[2] = 10            # pos_angl
    #     cube[3] = 10            # incl
    #     cube[4] = 10            # syt_vel
    #     cube[5] = 10            # vmax
    #     cube[6] = 10            # charac_rad
    #     return
    #
    # def loglike(cube, ndim, nparams):
    #
    #     model.set_parameters(*cube, flux_hd)
    #     model.velocity_map(psf, flux_ld, flux_hd, vel_model)
    #
    #     return np.array(np.log10(vel.data[flux_ld.mask]-model.vel_map[flux_ld.mask])-np.log10(errvel[flux_ld.mask])).sum()
    #
    # params = ['xcen', 'ycen', 'pa', 'incl', 'vs', 'vm', 'rd']
    # n_params = len(params)
    #
    # pymultinest.run(loglike, prior, n_params, outputfiles_basename='test_', resume=False, verbose=True)
    #
    # output = pymultinest.Analyzer(n_params=n_params, outputfiles_basename='test_')
    # stats = output.get_stats()


