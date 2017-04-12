#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt

from Model2D import Model2D
from PSF import PSF
import cap_mpfit as mpfit


def use_mpfit(psf, flux_ld, flux_hd, vel, errvel, params, model_name, slope=0, quiet=1):
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
    model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_hd)
    model.velocity_map(psf, flux_ld, flux_hd, model_name)

    def func_fit(p, fjac=None, data=None, err=None, vel_model=None, psf=None, flux_ld=None, flux_hd=None):

        xcen = p[0]
        ycen = p[1]
        pos_angl = p[2]
        incl = p[3]
        syst_vel = p[4]
        vmax = p[5]
        charac_rad = p[6]

        model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_hd)
        model.velocity_map(psf, flux_ld, flux_hd, vel_model)

        return [0, np.reshape((data.data[flux_ld.mask]-model.vel_map[flux_ld.mask])/err.data[flux_ld.mask], -1)]

    # PARINFOS
    p0 = [xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad]
    p0names = ['xcen', 'ycen', 'pos_angl', 'incl', 'syst_vel', 'vmax', 'charac_rad']
    parinfo = [{'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'parname': 0., 'step': 0.} for i in range(len(p0))]

    # xcen
    # parinfo[0]['fixed'] = 1
    # ycen
    # parinfo[1]['fixed'] = 1
    # Position angle
    parinfo[2]['limited'] = [1, 1]
    parinfo[2]['limits'] = [-180, 180]
    # Inclination
    # parinfo[3]['fixed'] = 1
    parinfo[3]['limited'] = [1, 1]
    parinfo[3]['limits'] = [5, 85]
    # syst vel
    parinfo[4]['limited'] = [1, 1]
    parinfo[4]['limits'] = [0, 1000]
    # vm
    parinfo[5]['limited'] = [1, 1]
    parinfo[5]['limits'] = [0, 1000]
    # characteristic radius
    # parinfo[6]['fixed'] = 1
    parinfo[6]['limited'] = [1, 1]
    parinfo[6]['limits'] = [1, charac_rad+10]

    for i in range(len(p0)):
        parinfo[i]['value'] = p0[i]
        parinfo[i]['parname'] = p0names[i]

    funckw = {'data': vel, 'flux_ld': flux_ld, 'flux_hd': flux_hd, 'err': errvel, 'vel_model': model_name, 'psf': psf}

    print('\nstart fit with mpfit')
    t1 = time.time()

    model_fit = mpfit.mpfit(func_fit, parinfo=parinfo, functkw=funckw, autoderivative=1, gtol=1e-10, ftol=1e-10, xtol=1e-10, quiet=quiet)

    t2 = time.time()
    print('fit done in: {} s'.format(t2-t1))

    print('fit status:', model_fit.status)
    print('Chi2R: {} DOF: {}'.format(model_fit.fnorm/model_fit.dof, model_fit.dof,))

    print('\nx    y    pa    incl    vs    vm    d')
    print('params: {}'.format(model_fit.params))
    print('from model : {}'.format(model.get_parameter(flux_hd)))

    # tools.write_fits(*model_fit.params, sig0, model.vel_map, 'validation/modv', chi2r=model_fit.fnorm/model_fit.dof, dof=model_fit.dof)
    # tools.write_fits(*model_fit.params, sig0, vel.data-model.vel_map, 'validation/resv', chi2r=model_fit.fnorm / model_fit.dof, dof=model_fit.dof)
    # ascii.write(model_fit.params, 'validation/fit_python.txt', names=['x', 'y', 'pa', 'incl', 'vs', 'vm', 'd'])

    plt.figure()
    plt.imshow(vel.data-model.vel_map, cmap='nipy_spectral')
    plt.colorbar()
    plt.show()