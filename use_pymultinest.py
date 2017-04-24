import numpy as np
import pymultinest
import time
import os
from Model2D import Model2D
from PSF import PSF
import tools
from astropy.io import ascii


def use_pymultinest(psf, flux_ld, flux_hd, vel, errvel, params, vel_model, slope=0, quiet=False):
    """

    :param PSF psf: 
    :param Image flux_ld: 
    :param Image flux_hd: 
    :param Image vel: 
    :param Image errvel: 
    :param Union[ndarray, Iterable] params: 
    :param vel_model: 
    :param float slope: 
    :param int quiet: 
    :return: 
    """

    gal, xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, sig0, fwhm, psfz, smooth = params

    model = Model2D(flux_ld, flux_hd, sig0, slope=slope)

    def prior(cube, ndim, nparams):
        cube[0] = cube[0] * 20 + xcen - 10      # prior between xcen - 5 ans xcen + 5
        cube[1] = cube[1] * 20 + ycen - 10      # prior between ycen - 5 ans ycen + 5
        cube[2] = cube[2] * 360 - 180           # pos angl between -180 and 180 degree
        cube[3] = cube[3] * 40 + incl - 20      # incl between incl - 20 and incl  + 20
        cube[4] = cube[4] * 1000 - 500          # sys vel between -500 and 500km/s
        cube[5] = cube[5] * 500                 # vmax between 0 and 500km/s
        cube[6] = cube[6] * 10 + charac_rad - 5 # charac rad between 1 and 15

    def loglike(cube, ndim, nparams):

        model.set_parameters(cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], flux_hd)
        model.velocity_map(psf, flux_ld, flux_hd, vel_model)

        chi2 = -(vel.data[flux_ld.mask] - model.vel_map[flux_ld.mask])**2/(2*errvel.data[flux_ld.mask]**2)

        return np.sum(chi2)

    params = ['xcen', 'ycen', 'pa', 'incl', 'vs', 'vm', 'rd']
    n_params = len(params)

    t1 = time.time()
    pymultinest.run(loglike, prior, n_params, outputfiles_basename='../test/1500317/multinest/output_', resume=False, verbose=quiet, max_iter=10000,
                    n_live_points=500)
    t2 = time.time()
    print('fit done in: {:6.2f} s'.format(t2-t1))

    output = pymultinest.Analyzer(n_params=n_params, outputfiles_basename='../test/1500317/multinest/output_')
    stats = output.get_mode_stats()
    bestfit = output.get_best_fit()

    model.set_parameters(*bestfit['parameters'], flux_hd)
    model.velocity_map(psf, flux_ld, flux_hd, vel_model)

    print('{0:^{width}}{1:^{width}}{2:^{width}}{3:^{width}}{4:^{width}}{5:^{width}}'
          '{6:^{width}}'.format('xcen', 'ycen', 'pa', 'incl', 'vs', 'vm', 'rd', width=12))
    print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}'
          '{4:^{width}.{prec}f}{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*bestfit['parameters'], width=12, prec=6))
    print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}'
          '{4:^{width}.{prec}f}{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*stats['modes'][0]['sigma'], width=12, prec=6))

    tools.write_fits(*bestfit['parameters'], sig0, model.vel_map, '../test/1500317/multinest/modv', mask=flux_ld.mask)
    tools.write_fits(*bestfit['parameters'], sig0, vel.data-model.vel_map, '../test/1500317/multinest/resv', mask=flux_ld.mask)
    ascii.write(np.array(bestfit['parameters']), '../test/1500317/multinest/fit_python.txt',
                names=['x', 'y', 'pa', 'incl', 'vs', 'vm', 'd'], overwrite=True)
