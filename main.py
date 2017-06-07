#!/usr/bin/env python
import argparse
from astropy.io import ascii, fits
import Tools.tools as tools
import Tools.velocity_model as vm
from Class.Images import Image, ImageOverSamp
from Class.PSF import PSF
from SubProcess.use_mpfit import use_mpfit
from SubProcess.use_pymultinest import use_pymultinest


def main(parser):
    parser.add_argument('model', help="model of the velocity field exp, flat, arctan etc")
    parser.add_argument('path', help="path to the directory where files are")
    parser.add_argument('input_txt', help="txt file which contain the initial parameters for the model")
    parser.add_argument('fits_ld', help="fits file of the flux of the galaxy in 'low resolution'")
    parser.add_argument('fits_vel', help="fits file of the velocity field in the same resolution as the flux fits")
    parser.add_argument('fits_evel', help="fits file of the error on the velocity field")
    parser.add_argument('-hd', '--high', dest="fits_hd", help="fits file of the flux of the galaxy in higher resolution")
    parser.add_argument('-v', '--verbose', default=False, action='store_true', dest='verbose', help="to see verbose of fit's method")
    parser.add_argument('-slope', '--slope', dest="slope", type=float, default=0., help="slope of the dispersion, default is 0")
    parser.add_argument('-psf', '--psf', dest='psf', default=None, help="fits of the psf of the low resolution flux, if None a gaussian is use")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mpfit', action='store_false', default=False, dest='mpfit_multinest', help="use chi square reduction method for fit")
    group.add_argument('--multinest', action='store_true', dest="mpfit_multinest", help="use a bayesian method for fit")
    parser.add_argument('-nbp', default=19000, type=int, help="number of calculated points by multinest, by default set to 19000 plus 1000 live point")
    parser.add_argument('-incfix', '--incfix', default=False, action='store_true', help="fix the inclination to which is in the txt file")
    parser.add_argument('-xfix', '--xfix', default=False, action='store_true', help="fix the position in x to which is in the txt file")
    parser.add_argument('-yfix', '--yfix', default=False, action='store_true', help="fix the position in y to which is in the txt file")
    args = parser.parse_args()

    print('\n entering in directory {}'.format(args.path.split('/')[-2]))

    params_file = tools.search_file(args.path, args.input_txt)
    params = ascii.read(params_file)[0]

    flux_ld_file = tools.search_file(args.path, args.fits_ld)
    flux_ld = Image(flux_ld_file)

    # If no high resolution image are given, we perform an interpolation of the flux distribution to an higher resolution
    if args.fits_hd:
        flux_hd_file = tools.search_file(args.path, args.fits_hd)
        flux_hd = Image(flux_hd_file)
        flux_hd.oversample = int(flux_hd.length / flux_ld.length)
        whd = '_whd'
    else:
        flux_hd = ImageOverSamp(flux_ld_file, params[7])
        flux_hd.oversample = int(flux_hd.length / flux_ld.length)
        whd = ''

    vel_file = tools.search_file(args.path, args.fits_vel)
    vel = Image(vel_file)

    evel_file = tools.search_file(args.path, args.fits_evel)
    errvel = Image(evel_file)

    if args.psf:
        psf_file = tools.search_file(args.path, args.psf)
        img_psf = fits.getdata(psf_file)
    else:
        img_psf = None
    print('\n all images found \n import the chosen module \n using {}x{} images'.format(vel.size[0], vel.size[1]))

    model_name = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}

    psf = PSF(flux_hd, img_psf, fwhm_ld=params[9], smooth=params[11])
    print(' using fwhm  {} pixels for HD with oversample {}'.format(psf.fwhm_f, flux_hd.oversample))

    # convolution and rebinnig HD image before calculation (after interpolation if needed)
    flux_hd.conv_inter_flux(psf)

    if args.mpfit_multinest is True:
        use_pymultinest(psf, flux_ld, flux_hd, vel, errvel, params, model_name[args.model], args.path, slope=args.slope, quiet=args.verbose, whd=whd,
                        incfix=args.incfix, xfix=args.xfix, yfix=args.yfix, nbp=args.nbp)
    else:
        if args.verbose:
            quiet = 0
        else:
            quiet = 1
        use_mpfit(psf, flux_ld, flux_hd, vel, errvel, params, model_name[args.model], args.path, slope=args.slope, quiet=quiet, whd=whd,
                  incfix=args.incfix, xfix=args.xfix, yfix=args.yfix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="\t(name not found) fit model to velocity field of galaxies. "
                                                 "\n\tFor more information see the help or refer to the git repository:"
                                                 "\n\thttps://github.com/Meirdrarel/batman"
                                                 "\n\tdeveloped on python 3.6 \n\t@uthor: Jérémy Dumoulin",
                                     formatter_class=argparse.RawTextHelpFormatter)
    main(parser)


