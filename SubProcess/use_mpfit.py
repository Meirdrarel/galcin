import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))
import time
import numpy as np
from astropy.io import ascii
import Tools.tools as tools
from Class.Model2D import Model2D
from Class.PSF import PSF
from Class.Images import Image, ImageOverSamp
import Tools.cap_mpfit as mpfit


def use_mpfit(psf, flux_ld, flux_hd, vel, errvel, params, model_name, path, slope=0, quiet=1, whd='', incfix = False, xfix=False, yfix=False):
    """

    This function define parameters and constraints for mpfit before to procede at the analysis. Write fits files of the best model and the difference with
    data.

    :param PSF psf:
    :param Image flux_ld:
    :param Image flux_hd:
    :param Image vel:
    :param Image errvel:
    :param ndarray params:
    :param func model_name:
    :param string path:
    :param float slope:
    :param Bool quiet:
    :param string whd:
    :param Bool incfix:
    :param Bool xfix:
    :param Bool yfix:
    """
    gal, xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, sig0, fwhm, psfz, smooth = params

    model = Model2D(flux_ld, flux_hd, sig0, slope=slope)
    model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_hd)

    def func_fit(p, fjac=None, data=None, err=None, vel_model=None, psf=None, flux_ld=None, flux_hd=None):
        """
        Function minimized by mpfit.

        Return (data-model)^2/err^2
        """
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
    if xfix:
        parinfo[0]['fixed'] = 1
    parinfo[0]['limited'] = [1, 1]
    parinfo[0]['limits'] = [xcen-5, xcen+5]
    # ycen
    if yfix:
        parinfo[1]['fixed'] = 1
    parinfo[1]['limited'] = [1, 1]
    parinfo[1]['limits'] = [ycen-5, ycen+5]
    # Position angle
    parinfo[2]['limited'] = [1, 1]
    parinfo[2]['limits'] = [-180, 180]
    # Inclination
    if incfix:
        parinfo[3]['fixed'] = 1
    parinfo[3]['limited'] = [1, 1]
    parinfo[3]['limits'] = [5, 85]
    # syst vel
    parinfo[4]['limited'] = [1, 1]
    parinfo[4]['limits'] = [-500, 500]
    # vm
    parinfo[5]['limited'] = [1, 1]
    parinfo[5]['limits'] = [0, 1000]
    # characteristic radius
    # parinfo[6]['fixed'] = 1
    parinfo[6]['limited'] = [1, 1]
    parinfo[6]['limits'] = [0.5, charac_rad+10]

    for i in range(len(p0)):
        parinfo[i]['value'] = p0[i]
        parinfo[i]['parname'] = p0names[i]

    funckw = {'data': vel, 'flux_ld': flux_ld, 'flux_hd': flux_hd, 'err': errvel, 'vel_model': model_name, 'psf': psf}

    print('\n start fit with mpfit')
    t1 = time.time()

    model_fit = mpfit.mpfit(func_fit, parinfo=parinfo, functkw=funckw, autoderivative=1, gtol=1e-10, ftol=1e-10, xtol=1e-10, quiet=quiet)

    t2 = time.time()
    print(' fit done in: {:6.2f} s\n'.format(t2-t1))

    # Print results on screen
    print(' fit status:', model_fit.status)
    print(' Chi2R: {} DOF: {}'.format(model_fit.fnorm/model_fit.dof, model_fit.dof,))

    print('', '-' * 81)
    print('{0:^{width}}{1:^{width}}{2:^{width}}{3:^{width}}{4:^{width}}'
          '{5:^{width}}{6:^{width}}'.format('xcen', 'ycen', 'pa', 'incl', 'vs', 'vm', 'rd', width=12))
    print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}{4:^{width}.{prec}f}'
          '{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*model_fit.params, width=12, prec=6))
    print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}{4:^{width}.{prec}f}'
          '{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*model_fit.perror, width=12, prec=6))
    print('', '-' * 81)

    # Write fits file
    dirname = 'mpfit'
    if incfix or xfix or yfix:
        dirname += '_'
    if xfix:
        dirname += 'x'
    if yfix:
        dirname += 'y'
    if incfix:
        dirname += 'i'
    if os.path.isdir(path+dirname) is False:
        os.makedirs(path+dirname)

    tools.write_fits(*model_fit.params, sig0, model.vel_map, path+'/mpfit/modv'+whd, chi2r=model_fit.fnorm/model_fit.dof, dof=model_fit.dof,
                     mask=flux_ld.mask)
    tools.write_fits(*model_fit.params, sig0, vel.data-model.vel_map, path+'/mpfit/resv'+whd, chi2r=model_fit.fnorm / model_fit.dof, dof=model_fit.dof,
                     mask=flux_ld.mask)
    tools.write_fits(*model_fit.params, sig0, model.vel_map_hd, path+'/mpfit/modv_hd'+whd, chi2r=model_fit.fnorm / model_fit.dof, dof=model_fit.dof,
                     oversample=1/flux_hd.oversample),

    model.vel_disp_map(flux_ld, flux_hd, psf)
    tools.write_fits(*model_fit.params, sig0, model.vel_disp, path + '/mpfit/modd' + whd, chi2r=model_fit.fnorm / model_fit.dof, dof=model_fit.dof,
                     mask=flux_ld.mask)
    # tools.write_fits(*model_fit.params, sig0, vel.data - model.vel_map, path + '/mpfit/resd' + whd, chi2r=model_fit.fnorm / model_fit.dof, dof=model_fit.dof,
    #                  mask=flux_ld.mask)

    param_write = np.append(np.append(model_fit.params, model_fit.fnorm), model_fit.dof)
    error = np.append(np.append(model_fit.perror, 0), 0)
    ascii.write(np.array([param_write, error]), path + '/mpfit/fit_python' + whd + '.txt',
                names=['x', 'y', 'pa', 'incl', 'vs', 'vm', 'rd', 'chi2', 'dof'], format='fixed_width', delimiter=None,
                formats={'x': '%10.6f', 'y': '%10.6f', 'pa': '%10.6f', 'incl': '%10.6f', 'vs': '%10.6f', 'vm': '%10.6f', 'rd': '%10.6f', 'chi2': '%10.6f',
                         'dof': '%10.6f'}, overwrite=True)

    if type(flux_hd) is ImageOverSamp:
        tools.write_fits(0, 0, 0, 0, 0, 0, 0, 0, flux_hd.data, path + '/mpfit/flux_hd_interpol')