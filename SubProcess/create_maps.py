#!/usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))
import numpy as np
from numpy.fft import fftshift, fft2, ifft2
import math
import Tools.velocity_model as vm
import Tools.flux_model as fm
import Tools.tools as tools
from Class.Images import Image, ImageOverSamp
from Class.PSF import PSF
from Class.Model2D import Model2D
import argparse
from astropy.io import ascii
import matplotlib.pyplot as plt


def main(parser):
    parser.add_argument('path', help='directory where will create the map', type=str)
    parser.add_argument('-fm',  default='flat', help='flux model', type=str)
    parser.add_argument('-vm', default='flat', help='velocity model', type=str)
    parser.add_argument('-pa', default=0, type=float, help='position angel from the north to the left of the galaxy')
    parser.add_argument('-incl', default=45, type=float, help='inclination of th galactic plane')
    parser.add_argument('-vs', default=0, type=float, help='systemic velocity')
    parser.add_argument('-vmax', default=200, type=float, help='characteristic velocity of the model')
    parser.add_argument('-rdv', default=4, type=float, help='characteristic radius of the velocity model')
    parser.add_argument('-rdf', default=6, type=float, help='characteristic radius of the flux model')
    parser.add_argument('-sig0', default=40, type=float, help='velocity dispersion')
    parser.add_argument('-fwhm', default=3.5, type=float, help='fwhm of a gaussian psf')
    parser.add_argument('-smooth', default=0, type=float, help='fwhm for smooth images')
    parser.add_argument('-rt', default=8, type=float, help='truncated radius after which the flux is set to 0')
    parser.add_argument('-osamp', default=5, type=int, help='oversampling to compute high definition')
    parser.add_argument('-center', nargs='+', default=(15, 15), type=float, help='center of the galaxy')
    parser.add_argument('-size', nargs='+', default=(30, 30), type=int, help='size of images in low resolution')
    parser.add_argument('-prefix', type=str, help='prefix add to filename')
    parser.add_argument('-suffix', type=str, help='suffix add to filename')
    parser.add_argument('-slope', type=float, help="")
    args = parser.parse_args()

    fm_list = {'exp': fm.exponential_disk_intensity, 'flat': fm.flat_disk_intensity}
    vm_list = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

    center_bright = 2000
    # oversampling paramters#
    rdv = args.rdv * args.osamp
    rdf = args.rdf * args.osamp
    fwhm = args.fwhm * args.osamp
    smooth = args.smooth * args.osamp
    new_xcen = (args.center[0]+0.5)*args.osamp-0.5
    new_ycen = (args.center[1]+0.5)*args.osamp-0.5
    im_size = np.array(args.size) * args.osamp
    rtrunc = args.rt * args.osamp
    print('\n create array {}x{}'.format(*im_size))

    def conv_psf(data, fwhm):
        # (2*fwhm) is considered to avoid the boundary effect after the convolution
        size = np.array([2 * fwhm, 2 * fwhm])
        ideal_size = 2 ** np.ceil(np.log(im_size + 2 * size) / math.log(2))
        # to prevent any problems in the future, size has been forced to be an array of int
        size = np.array((ideal_size - im_size) / 2, dtype=int)

        y, x = np.indices((im_size[0] + 2 * size[0], im_size[1] + 2 * size[1]))
        sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
        psf = (1. / (2 * math.pi * sigma ** 2)) * np.exp(-((x - im_size[1] / 2 - size[1]) ** 2 + (y - im_size[0] / 2 - size[0]) ** 2) / (2.0 * sigma ** 2))

        # normalization in order to ensure flux conservation
        psf /= psf.sum()

        data2 = np.zeros((data.shape[0] + 2 * size[0], data.shape[1] + 2 * size[1]))
        data2[size[0]:data.shape[0] + size[0], size[1]:data.shape[1] + size[1]] = data

        psf_fft2 = fft2(fftshift(psf))

        data_conv = ifft2(fft2(data2) * psf_fft2)
        data_conv = data_conv[size[0]:data.shape[0] + size[0], size[1]:data.shape[1] + size[1]].real

        return data_conv

    # create flux maps
    flux = fm_list[args.fm](new_xcen, new_ycen, args.pa, args.incl, rdf, center_bright, rtrunc, im_size)
    # fluxw = np.copy(flux)

    flux_rebin = tools.rebin_data(conv_psf(flux, np.sqrt(fwhm**2+args.smooth**2)), args.osamp)

    # create velocity maps
    flux_ld = Image(flux_rebin, mask=flux_rebin>0.1)
    flux_hd = ImageOverSamp(flux_rebin, rdv, oversamp=args.osamp)
    psf = PSF(flux_hd, fwhm_ld=args.fwhm, smooth=args.smooth)
    flux_hd.conv_inter_flux(psf)

    model = Model2D(flux_ld, flux_hd, args.sig0, slope=args.slope)
    model.set_parameters(args.center[0], args.center[1], args.pa, args.incl, args.vs, args.vmax, args.rdv, flux_hd)
    model.velocity_map(psf, flux_ld, flux_hd, vm_list[args.vm])

    # flux[np.where(flux < 0.1)] = 0
    print(' write flux hd')
    tools.write_fits(new_xcen, new_ycen, args.pa, args.incl, args.vs, args.vmax, rdv, 0, psf.convolution(flux), args.path +'flux_HD')

    # flux_rebin[np.where(flux_rebin < 0.1)] = 0
    print(' write flux ld')
    tools.write_fits(*args.center, args.pa, args.incl, args.vs, args.vmax, args.rdv, 0, flux_rebin, args.path +'flux')

    vel_hd = psf.convolution(model.vel_map_hd)
    print(' write vel map hd')
    tools.write_fits(new_xcen, new_ycen, args.pa, args.incl, args.vs, args.vmax, rdv, 0, vel_hd, args.path +'modelV_HD')

    print(' write vel map')
    tools.write_fits(args.center[0], args.center[1], args.pa, args.incl, args.vs, args.vm, args.rdv, 0, tools.rebin_data(vel_hd, args.osamp),
                     args.path + 'modelV_full')
    model.vel_map[flux_rebin < 0.1] = float('nan')
    tools.write_fits(args.center[0], args.center[1], args.pa, args.incl, args.vs, args.vm, args.rdv, 0, model.vel_map, args.path +'modelV')

    # create dispersion and error maps
    evel = np.ones(args.size)
    evel[flux_rebin < 0.1] = float('nan')
    tools.write_fits(args.center[0], args.center[1], args.pa, args.incl, args.vs, args.vm, args.rdv, 0, evel, args.path+'evel')

    sig0 = np.ones(args.size)*args.sig0
    sig0[flux_rebin < 0.1] = float('nan')
    tools.write_fits(args.center[0], args.center[1], args.pa, args.incl, args.vs, args.vm, args.rdv, 0, sig0, args.path+'disp')

    # write ascii file with model parameters
    print(' write parameter file\n')
    params = np.array([args.center[0], args.center[1], args.pa, args.incl, args.vs, args.vmax, rdv/args.osamp, args.sig0, fwhm/args.osamp, smooth,
                       args.osamp, args.rt])
    names = ['x', 'y', 'pa', 'incl', 'vs', 'vm', 'rd', 'sig0', 'fwhm', 'smooth', 'oversample', 'rtrunc']
    formats = {'x': '%10.1f', 'y': '%10.1f', 'pa': '%10.1f', 'incl': '%10.1f', 'vs': '%10.1f', 'vm': '%10.1f', 'rd': '%10.1f', 'sig0': '%10.1f',
               'fwhm': '%10.1f', 'smooth': '%10.1f', 'oversample': '%10.1f', 'rtrunc': '%10.1f'}
    ascii.write(params, args.path+'param_model.txt', names=names, delimiter=None, formats=formats, format='fixed_width', overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="\tCreate flux map and velocity map using (main prog name)'s model"
                                                 "\n\tFor more information see the help or refer to the git repository:"
                                                 "\n\thttps://github.com/Meirdrarel/batman"
                                                 "\n\tdeveloped on python 3.6",
                                     formatter_class=argparse.RawTextHelpFormatter)
    main(parser)
