import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))
import numpy as np
import pymultinest
import time
from Class.Model2D import Model2D
from Class.PSF import PSF
from Class.Images import Image, ImageOverSamp
import Tools.tools as tools
from astropy.io import ascii
import matplotlib.pyplot as plt


def use_pymultinest(psf, flux_ld, flux_hd, vel, errvel, params, vel_model, path, rank=0, slope=0, quiet=False,
                    whd='', incfix=False, xfix=False, yfix=False, nbp=19000):
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
        if xfix:
            cube[0] = cube[0] * 2 - 1 + xcen         # prior between xcen - 1 ans xcen + 1
        else:
            cube[0] = cube[0] * 10 - 5 + xcen

        if yfix:
            cube[1] = cube[1] * 2 - 1 + ycen
        else:
            cube[1] = cube[1] * 10 - 5 + ycen         # prior between ycen - 5 ans ycen + 5

        if incfix:
            cube[3] = cube[3] * 2 + incl - 1       # incl between incl - 1 and incl  + 1 degree
        else:
            cube[3] = cube[3] * 80 + 5               # incl between 5 and 85 degree

        cube[2] = cube[2] * 360 - 180            # pos angl between -180 and 180 degree
        cube[4] = cube[4] * 200 - 100           # sys vel between -500 and 500km/s
        cube[5] = cube[5] * 500                  # vmax between 0 and 500km/s
        cube[6] = cube[6] * 14 + 1               # charac rad between 1 and 15

    def loglike(cube, ndim, nparams):

        model.set_parameters(cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6], flux_hd)
        model.velocity_map(psf, flux_ld, flux_hd, vel_model)

        chi2 = -(vel.data[flux_ld.mask] - model.vel_map[flux_ld.mask])**2/(2*errvel.data[flux_ld.mask]**2)

        return np.sum(chi2)

    params = ['xcen', 'ycen', 'pa', 'incl', 'vs', 'vm', 'rd']
    n_params = len(params)

    if rank == 0:
        dirname = 'multinest'
        if incfix or xfix or yfix:
            dirname += '_'
        if xfix:
            dirname += 'x'
        if yfix:
            dirname += 'y'
        if incfix:
            dirname += 'i'
        if os.path.isdir(path + dirname) is False:
            os.makedirs(path + dirname)

    if rank == 0:
        t1 = time.time()

    pymultinest.run(loglike, prior, n_params, outputfiles_basename=path+'/multinest/output'+whd+'_', resume=False, verbose=quiet, max_iter=nbp,
                    n_live_points=1000)

    if rank == 0:

        t2 = time.time()

        print('\n fit done in: {:6.2f} s \n'.format(t2-t1))

        output = pymultinest.Analyzer(n_params=n_params, outputfiles_basename=path+'/multinest/output'+whd+'_')
        stats = output.get_mode_stats()

        bestfit = stats['modes'][0]['maximum']
        model.set_parameters(*bestfit, flux_hd)

        print('', '-'*81)
        print('{0:^{width}}{1:^{width}}{2:^{width}}{3:^{width}}{4:^{width}}{5:^{width}}'
              '{6:^{width}}'.format('xcen', 'ycen', 'pa', 'incl', 'vs', 'vm', 'rd', width=12))
        print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}'
              '{4:^{width}.{prec}f}{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*bestfit, width=12, prec=6))
        print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}'
              '{4:^{width}.{prec}f}{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*stats['modes'][0]['sigma'], width=12, prec=6))
        print('', '-' * 81)

        if type(flux_hd) is ImageOverSamp:
            tools.write_fits(0, 0, 0, 0, 0, 0, 0, 0, flux_hd.data, path + '/multinest/flux_hd_interpol')

        model.set_parameters(*bestfit, flux_hd)
        model.velocity_map(psf, flux_ld, flux_hd, vel_model)

        tools.write_fits(*bestfit, sig0, model.vel_map, path+'/multinest/modv'+whd, mask=flux_ld.mask)
        tools.write_fits(*bestfit, sig0, model.vel_map, path + '/multinest/modv_full' + whd)
        tools.write_fits(*bestfit, sig0, vel.data-model.vel_map, path+'/multinest/resv'+whd, mask=flux_ld.mask)
        tools.write_fits(*bestfit, sig0, model.vel_map_hd, path + '/multinest/modv_hd' + whd, oversample=1 / flux_hd.oversample)
        model.vel_disp_map(flux_ld, flux_hd, psf)
        tools.write_fits(*bestfit, sig0, model.vel_disp, path + '/mpfit/modd' + whd, mask=flux_ld.mask)
        ascii.write(np.array([bestfit, stats['modes'][0]['sigma']]), path+'/multinest/params_fit'+whd+'.txt',
                    names=['x', 'y', 'pa', 'incl', 'vs', 'vm', 'rd'], format='fixed_width', delimiter=None,
                    formats={'x': '%.6f', 'y': '%.6f', 'pa': '%.6f', 'incl': '%.6f', 'vs': '%.6f', 'vm': '%.6f', 'rd': '%.6f'}, overwrite=True)

        all_stats = output.get_stats()
        plot = True
        for i in range(n_params):
            if all_stats['marginals'][i]['sigma'] == 0:
                plot = False

        if plot:
            parameters = ['x', 'y', 'pa', 'incl', 'vs', 'vm', 'rd']
            p = pymultinest.PlotMarginalModes(output)
            plt.figure(figsize=(5 * n_params, 5 * n_params))
            # plt.subplots_adjust(wspace=0, hspace=0)
            for i in range(n_params):
                plt.subplot(n_params, n_params, n_params * i + i + 1)
                p.plot_marginal(i, with_ellipses=False, with_points=False, grid_points=50)
                plt.ylabel("Probability")
                plt.xlabel(parameters[i])

                for j in range(i):
                    plt.subplot(n_params, n_params, n_params * j + i + 1)
                    # plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
                    p.plot_conditional(i, j, with_ellipses=False, with_points=True, grid_points=30)
                    plt.xlabel(parameters[i])
                    plt.ylabel(parameters[j])

            plt.savefig(path + '/multinest/proba'+whd+'.pdf')
        else:
            print('\n There is impossible to plot the statistics, one or more parameters has a sigma null \n')
