#!usr/bin/env python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))
import Tools.velocity_model as vm
import Tools.tools as tools
from Class.Images import Image, ImageOverSamp
from Class.PSF import PSF
import argparse
import numpy as np
import math
from astropy.io import ascii


parser = argparse.ArgumentParser()
parser.add_argument('path', help='directory where will create the map', type=str)
parser.add_argument('flux', type=str)
parser.add_argument('-vm', default='exp', help='velocity model', type=str)
parser.add_argument('-fm', default='flat', help='flux model', type=str)
parser.add_argument('-full', default=False, action='store_true', help='Full map ')
args = parser.parse_args()


params = ascii.read(args.path+'param_model.txt')
##################
#MODEL PARAMETERS#
##################
pos_angl = np.array(params['pa'])[0]
incl = np.array(params['incl'])[0]
syst_vel = np.array(params['vs'])[0]
vmax = np.array(params['vm'])[0]
charac_rad = np.array(params['rd'])[0]
rtrunc = np.array(params['rtrunc'])[0]
sig0 = np.array(params['sig0'])[0]
center_bright = 2000
oversample = np.array(params['oversample'])[0]
fwhm = np.array(params['fwhm'])[0]
smooth = np.array(params['smooth'])[0]
ycen = np.array(params['y'])[0]
xcen = np.array(params['x'])[0]
##################


#Input flux
flux = Image(args.path+args.flux)
flux_hd = ImageOverSamp(args.path+args.flux, charac_rad)
psf = PSF(flux_hd, fwhm_ld=fwhm, smooth=smooth)
flux_hd.conv_inter_flux(psf)

vm_list = {'exp': vm.exponential_velocity, 'flat': vm.flat_velocity, 'arctan': vm.arctan_velocity}
xcen_b = (xcen+0.5)*flux_hd.oversample-0.5
ycen_b = (ycen+0.5)*flux_hd.oversample-0.5
radius, theta = tools.sky_coord_to_galactic(xcen_b, ycen_b, pos_angl, incl, im_size=np.array(flux_hd.size, dtype=int))

#model velocity
vr = vm_list[args.vm](radius, charac_rad*flux_hd.oversample, vmax)
# Calculation of the velocity field
vel = vr * math.sin(math.radians(incl)) * theta + syst_vel
tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, 0, vel, args.path+'modelV')
# vel_times_flux = tools.rebin_data(psf.convolution(flux_hd.data * vel), flux_hd.oversample)
vel_times_flux = tools.rebin_data(psf.convolution(flux_hd.data * vel), flux_hd.oversample)
vel_map = np.zeros(flux.data.shape)

vel_map[flux.mask] = vel_times_flux[flux.mask] / flux_hd.data_rebin[flux.mask]

vel_map[np.where(vel_map == 0)] = float('NaN')
tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, 0, vel_map, args.path+'modelV_python')

tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, 0, np.ones(flux.data.shape), args.path+'evel')
tools.write_fits(xcen, ycen, pos_angl, incl, syst_vel, vmax, charac_rad, 0, np.ones(flux.data.shape)*sig0, args.path+'disp')
