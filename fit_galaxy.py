#!/usr/bin/env python

# Pyhton library
import sys
import os
import time
import numpy as np
from astropy.io import fits, ascii
import matplotlib.pyplot as plt

# Program files
from model_2D import Model2D
from PSF import PSF
from Images import Image, ImageOverSamp
import velocity_model as vm
import cap_mpfit as mpfit


def main(argv):
    """

    """

    # # HANDLE OF PROGRAMS ARGUMENTS
    # try:
    #     model_argv = argv[0]
    #     folder = argv[1]
    #     input_param = argv[2]
    #     options = str(argv[3])
    # except IndexError:
    #     print('Arguments error, use --help for more informations')
    #     sys.exit()
    #
    # if argv[4]:
    #     slope = float(argv[4])
    # else:
    #     slope = 0.
    #
    # ADD NEW MODEL IN THIS DICTIONARY:
    model_name = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

    # # OPEN FILE PARAMETER
    # try:
    #     param_fit = ascii.read(folder+input_param)
    # except FileNotFoundError:
    #     print('File not found')
    #     sys.exit()
    #
    # print('\nparameter file found')
    # gal, xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, sig0, fwhm, psfz, smooth = param_fit[0]
    #
    # filename = folder+options+str(gal)
    # flux_map = Image(filename+'_flux_OII3726.fits')
    # flux_map_over = ImageOverSamp(filename+'_flux_OII3726.fits', charac_rad)
    # vel_map = Image(filename+'_vel.fits')
    # err_vel = Image(filename+'_evel.fits')
    # img_psf = None
    # print('images imported successfully')

    gal, xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, sig0, fwhm, psfz, smooth = ascii.read('validation/param_model.txt')[0]
    flux_map = Image('validation/flux2.fits')
    flux_map_over = ImageOverSamp('validation/flux2.fits', charac_rad)
    vel_map = Image('validation/modelV_python.fits')
    err_vel = Image('validation/evel_120ones.fits')
    img_psf = None
    slope = 0
    model_argv = 'exp'

    psf = PSF(flux_map_over, img_psf, fwhm=np.sqrt(fwhm**2+smooth**2))
    model = Model2D(flux_map, flux_map_over, sig0, slope=slope)
    model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_map_over)

    def func_fit(p, fjac=None, data=None, err=None, vel_model=None, psf=None, flux_ld=None, flux_hd=None,):

        xcen = p[0]
        ycen = p[1]
        pos_angl = p[2]
        incl = p[3]
        syst_vel = p[4]
        vmax = p[5]
        charac_rad = p[6]

        model.set_parameters(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, flux_hd)
        model.velocity_map(psf, flux_ld, flux_hd, vel_model)

        return [0, np.reshape((vel_map.data[flux_ld.mask]-model.vel_map[flux_ld.mask])/err.data[flux_ld.mask], -1)]

    # PARINFOS
    p0 = [xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad]
    p0names = ['xcen', 'ycen', 'pos_angl', 'incl', 'syst_vel', 'vmax', 'charac_rad']
    parinfo=[{'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'parname': 0., 'step': 0.} for i in range(len(p0))]

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
    parinfo[6]['limited'] = [1, 1]
    parinfo[6]['limits'] = [1, charac_rad+10]

    for i in range(len(p0)):
        parinfo[i]['value'] = p0[i]
        parinfo[i]['parname'] = p0names[i]

    funckw = {'data': vel_map, 'flux_ld': flux_map, 'flux_hd': flux_map_over, 'err': err_vel, 'vel_model': model_name[model_argv], 'psf': psf}

    print('\nstart fit with mpfit')
    t1 = time.time()
    model_fit = mpfit.mpfit(func_fit, parinfo=parinfo, functkw=funckw, autoderivative=1, gtol=1e-10, ftol=1e-5, xtol=1e-10, quiet=0)
    t2 = time.time()
    print('fit done in: {} s'.format(t2-t1))
    print('fit status:', model_fit.status)
    print('Chi2: {} DOF: {} Chi2R: {}'.format(model_fit.fnorm, model_fit.dof, model_fit.fnorm/model_fit.dof))

    print('\nx    y    pa    incl    vs    vm    d')
    print('params: {}'.format(model_fit.params))
    print('from model : {}'.format(model.get_parameter(flux_map_over)))

    model.write_fits('validation/modv_python', model.vel_map, flux_map, flux_map_over, model_fit.fnorm/model_fit.dof)
    model.write_fits('validation/resv_python', vel_map.data-model.vel_map, flux_map, flux_map_over, model_fit.fnorm/model_fit.dof)
    ascii.write(model_fit.params, 'validation/fit_python.txt', names=['x', 'y', 'pa', 'incl', 'vs', 'vm', 'd'])

    # im_show = vel_map.data - model.vel_map
    # im_show[im_show == 0] = float('nan')
    # plt.figure(1)
    # plt.imshow(im_show, cmap='nipy_spectral')
    # plt.colorbar()
    # plt.figure(2)
    # plt.imshow(vel_map.data, cmap='nipy_spectral')
    # plt.colorbar()
    # plt.figure(3)
    # plt.imshow(model.vel_map, cmap='nipy_spectral')
    # plt.colorbar()
    # plt.show()

if __name__ == '__main__':
    main(sys.argv[1:])

# Bout de code interressant:
    # UTILISER ARGPARS !!!!
    # permet de passer des argument en option etc ...

    # try:
    #     opts, args = sys.getopt.getopt(argv, "hg:d", ["help", "grammar="])
    # except sys.getopt.GetoptError:
    #     usage()
    #     sys.exit(2)
