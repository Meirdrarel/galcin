#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ipdb
import time
from math import *
import numpy as np
import scipy as sp
from scipy.special import i0, i1, k0, k1
from scipy import constants as cst
from scipy.interpolate import interp1d
import matplotlib as mpl
from matplotlib import pyplot as plt
import pyfits as pf
#from numpy.fft import fft, ifft, fft2, ifft2, fftshift
from scipy.fftpack import fft, ifft, fft2, ifft2, fftshift
from pyfftw.interfaces.numpy_fft import fft, ifft, fft2, ifft2, fftshift, rfftn, irfftn, rfft, irfft, rfft2, irfft2


def exponential_disk_intensity_2D(b0, rd, rtrunc, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True):
    """Function that computes the intensity map of an exonential disk

    Parameters
    ----------
    b0: flt
        central surface brightness (erg/s/arcsec2)
    rd: flt
        disk scalelength (arcsec)
    rtrunc: flt
        truncation radius (arcsec) after which the flux is null
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordiantes of the center (arcsec)
    pa: flt
        position angle of the major axis (degree)
    incl: flt
        inclination of the disk (degree)
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec), res should be at least n x pixel size
    plot: bool
        keyword to show a figure of the result

    """

    nelement = np.int(np.ceil(2 * rlast / res))  # number of spatial sampling elements in rlast
    [y, x] = np.indices((nelement, nelement)) * res  # index matrices
    den = (y - ycen) * cos(radians(pa)) - (x - xcen) * sin(radians(pa))
    num = - (x - xcen) * cos(radians(pa)) - (y - ycen) * sin(radians(pa))
    r = (den ** 2 + (num / cos(radians(incl))) ** 2) ** 0.5  # radius in the galaxy plane

    # Calculation of the surface brightness in the plane of the galaxy
    if rd !=0:
        b = b0 * np.exp(- np.abs(r) / rd) * res ** 2
    else:
        b = b0 * np.exp(0 * r) * res ** 2

    b[np.where(r > rtrunc)] = 0.
    
    # Display of the result
    if plot is True:
        plt.imshow(b, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return b


def exponential_disk_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True):
    """Function that computes the velocity field for an exonential disk

    Parameters
    ----------
    vd: flt
        Maximum rotation velocity (km/s)
    rt: flt
        radius at wich the maximum velocity is reached (arcsec)
    vs: flt
        systemic velocity (km/s)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordinates of the center (arcsec)
    pa: flt
        position angle of the major axis (degree)
    incl: flt
        inclination of the disk (degree)
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec), res should be at least n x pixel size
    plot: bool
        keyword to show a figure of the result

    """

    nelement = np.int(np.ceil(2 * rlast / res))  # number of spatial sampling elements in rlast
    [y, x] = np.indices((nelement, nelement)) * res  # index matrices
    den = (y - ycen) * cos(radians(pa)) - (x - xcen) * sin(radians(pa))
    num = - (x - xcen) * cos(radians(pa)) - (y - ycen) * sin(radians(pa))
    tpsi = num * 1.
    tpsi[np.where(den != 0)] /= den[np.where(den != 0)]  # to avoid a NaN at the center
    den2 = cos(radians(incl)) ** 2 + tpsi ** 2
    sg = np.sign(den)  # signe
    ctheta = sg * (cos(radians(incl)) ** 2 / den2) ** 0.5  # azimuth in galaxy plane
    r = (den ** 2 + (num / cos(radians(incl))) ** 2) ** 0.5  # radius in the galaxy plane

    # Calculation of the rotational velocity in the plane of the galaxy
    rd = rt / 2.15  # disk scale length
    vr = r * 1.
    # np.where(r != 0) to avoid a NaN at the center
    q = np.where(r != 0)
    vr[q] = r[q] / rd * vd / 0.88 * np.sqrt(i0(0.5 * r[q] / rd) * k0(0.5 * r[q] / rd) 
                                            - i1(0.5 * r[q] / rd) * k1(0.5 * r[q] / rd))

    # Calculation of the velocity field
    v = vr * sin(radians(incl)) * ctheta + vs

    # Display of the result
    if plot is True:
        plt.imshow(v, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return v

def flat_model_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True):
    """Function that computes the velocity field for an exonential disk
        
    Parameters
    ----------
    vd: flt
        Maximum rotation velocity at the plateau (km/s)
    rt: flt
        tunover radius (arcsec)
    vs: flt
        systemic velocity (km/s)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordinates of the center (arcsec)
    pa: flt
        position angle of the major axis (degree)
    incl: flt
        inclination of the disk (degree)
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec), res should be at least n x pixel size
    plot: bool
        keyword to show a figure of the result
        
        """
    
    nelement = np.int(np.ceil(2 * rlast / res))  # number of spatial sampling elements in rlast
    [y, x] = np.indices((nelement, nelement)) * res  # index matrices
    den = (y - ycen) * cos(radians(pa)) - (x - xcen) * sin(radians(pa))
    num = - (x - xcen) * cos(radians(pa)) - (y - ycen) * sin(radians(pa))
    tpsi = num * 1.
    tpsi[np.where(den != 0)] /= den[np.where(den != 0)]  # to avoid a NaN at the center
    den2 = cos(radians(incl)) ** 2 + tpsi ** 2
    sg = np.sign(den)  # signe
    ctheta = sg * (cos(radians(incl)) ** 2 / den2) ** 0.5  # azimuth in galaxy plane
    r = (den ** 2 + (num / cos(radians(incl))) ** 2) ** 0.5  # radius in the galaxy plane
    
    #Rotation curve model flat. It doesn't describe any classical mass distribution, but can describe correctly a number of observed rotation curves of local galaxies, in particular those reaching the plateau (from appendix A5.3 Benoît et al 2009)
    #        V(r)=Vd * r/rt    for r<rt
    #        V(r)=Vd           for r>rt
    
    vr=r * 1
    # np.where(r != 0) to avoid a NaN at the center
    q = np.where(r != 0)
    sg = np.sign(r)  # sign

    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if np.abs(r[i,j]) < rt:
                vr[i,j] = vd * r[i,j] / rt
            else: vr[i,j] = sg[i,j] * vd

    # Calculation of the velocity field
    v = vr * sin(radians(incl)) * ctheta + vs

#    ipdb.set_trace()

    # Display of the result
    if plot is True:
        plt.imshow(v, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return v

def arctangent_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True):
    """Function that computes the velocity field for an exonential disk
        
    Parameters
    ----------
    vd: flt
        Maximum rotation velocity (km/s)
    rt: flt
        transition radius at which the velocity reaches 70 per cent of vd (arcsec)
    vs: flt
        systemic velocity (km/s)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordinates of the center (arcsec)
    pa: flt
        position angle of the major axis (degree)
    incl: flt
        inclination of the disk (degree)
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec), res should be at least n x pixel size
    plot: bool
        keyword to show a figure of the result
        
        """
    
    nelement = np.int(np.ceil(2 * rlast / res))  # number of spatial sampling elements in rlast
    [y, x] = np.indices((nelement, nelement)) * res  # index matrices
    den = (y - ycen) * cos(radians(pa)) - (x - xcen) * sin(radians(pa))
    num = - (x - xcen) * cos(radians(pa)) - (y - ycen) * sin(radians(pa))
    tpsi = num * 1.
    tpsi[np.where(den != 0)] /= den[np.where(den != 0)]  # to avoid a NaN at the center
    den2 = cos(radians(incl)) ** 2 + tpsi ** 2
    sg = np.sign(den)  # signe
    ctheta = sg * (cos(radians(incl)) ** 2 / den2) ** 0.5  # azimuth in galaxy plane
    r = (den ** 2 + (num / cos(radians(incl))) ** 2) ** 0.5  # radius in the galaxy plane
    
    # Rotation curve model: arctangent function. It is used in Puech at al 2008. Since the maximum velocity is reached asymptotically for infinite radius, the transition radius rt is defined as the radius for which the velocity reaches 70 per cent of the asymptotic velocity Vt.
    #        V(r)=Vt * 2/ pi arctan((2 * r)/rt)
    
    
    vr=r * 1
    # np.where(r != 0) to avoid a NaN at the center
    q = np.where(r != 0)
    sg = np.sign(r)  # sign
    
    vr= vd * (2. / pi) * np.arctan((2. * r)/ rt)


    # Calculation of the velocity field
    v = vr * sin(radians(incl)) * ctheta + vs
    
    # Display of the result
    if plot is True:
        plt.imshow(v, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return v




def linear_velocitydispersion_2D(sig0, slope, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True):
    """Function that computes the velocity dispersion map with a linearly decreasing profile

    Parameters
    ----------
    sig0: flt
        velocity dispersion (km/s)
    slope: flt
        slope of the velocity dispersion profile (km/s / arcsec)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordinates of the center (arcsec)
    pa: flt
        position angle of the major axis (degree)
    incl: flt
        inclination of the disk (degree)
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec), res should be at least n x pixel size
    plot: bool
        keyword to show a figure of the result

    """

    nelement = np.int(np.ceil(2 * rlast / res))  # number of spatial sampling elements in rlast
    [y, x] = np.indices((nelement, nelement)) * res  # index matrices
    den = (y - ycen) * cos(radians(pa)) - (x - xcen) * sin(radians(pa))
    num = - (x - xcen) * cos(radians(pa)) - (y - ycen) * sin(radians(pa))
    r = (den ** 2 + (num / cos(radians(incl))) ** 2) ** 0.5  # radius in the galaxy plane

    # Calculation of the velocity dispersion
    sig = sig0 + slope * np.abs(r)
    sig[np.where(sig <= 0)] = 0

    # Display of the result
    if plot is True:
        plt.imshow(sig, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return sig


def psf_gaussian_convolution_2D(cube, fwhm, xcen, ycen, rlast=10, res=0.04):
    """Function that computes the convolution of a datacube with a PSF 2D Gaussian function
        
    Parameters
    ----------
    cube: numpy 3D array
        high resolution datacube that has to be convolved with the PSF
    fwhm: flt
        full width at half maximum of the Gaussian function which represents the PSF (arcsec)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordinates of the center (arcsec)
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec), res should be at least n x pixel size       
        """

#   CREATION OF THE 2D GAUSSIAN FUNCTION
    nelement = np.int(np.ceil(2 * rlast / res))  # number of spatial sampling elements in rlast
    fwhmind = np.int(np.ceil(fwhm / res))       # number of spatial sampling elements in fwhm
    s = 2 * fwhmind         #(2*fwhmind) is considered to avoid the boundary effect after the convolution.
    
    ideal_size = 2 ** ceil(log(nelement + 2 * s) / log(2))
    s = (ideal_size - nelement) / 2
    
    [y, x] = np.indices((nelement + 2 * s, nelement + 2 * s)) * res  # index matrices

    sigma = fwhm / (2 * sqrt(2*log(2)))
    psf = (1./(2*pi*sigma**2)) * np.exp( -((x - xcen - s*res)**2 + (y - ycen - s*res)**2) / (2.0 * sigma ** 2) )
    psf /= psf.sum()        # normalization in order to ensure flux conservation

#   SPATIAL CONVOLUTION
    psf_sh = fftshift(psf)
    cube2 = np.zeros((cube.shape[0], cube.shape[1] + 2 * s, cube.shape[2] + 2 * s))
    cube2[:, s:cube.shape[1] + s, s:cube.shape[2] + s] = cube
#    cubeconv=ifft2(fft2(cube2) * fft2(psf_sh.reshape(1, psf.shape[0], psf.shape[1])))

    cubeconv=irfft2(rfft2(cube2) * rfft2(psf_sh.reshape(1, psf.shape[0], psf.shape[1])))
    
    cubeconv=cubeconv.real
    cubeconv=cubeconv[:, s:cube.shape[1]+s, s:cube.shape[2]+s]
    
    return cubeconv

def spectral_convolution(cube, kernel, lbda, lrange=16.8, lres=0.15):
    """Function that computes the convolution of a datacube with a Gaussian function that represent the intrinsic spectral resolution of the GRISM (linked to the number of grooves)

    Parameters
    ----------
    cube: numpy 3D array
        high resolution datacube after spatial smoothing
    kernel: flt
        FWHM in Angstrom of the spectral PSF (assumed to be Gaussian)
    lbda:flt
        central wavelength of model spectral range (A)
    lrange:flt
        spectral range in A
    lres: flt
        spectral resolution of the high resolution data (A)
    """
#   CREATION OF THE GAUSSIAN FUNCTION

    lsize = np.int(np.round( lrange / lres))        # number of spectral sampling
    kernelind = np.int(np.ceil(kernel / lres))       # number of spatial sampling elements in kernel
    ls = 2 * kernelind                      #(2*fwhmind) is considered to avoid the boundary effect after the convolution.
    
    ideal_size = 2 ** ceil(log(lsize + 2 * ls) / log(2))
    ls = (ideal_size - lsize) / 2
    
    lindex = (np.arange(lsize + 2 * ls) - (lsize + 2 * ls) / 2) * lres    #spectral sampling elements
#    vind = (lindex / lbda) * cst.c *1e-3              #velocity sampling elements

    
    sigma = kernel / (2 * sqrt(2 * log(2)))
    psfgrism = (1 / (sigma * sqrt(2 * pi))) * np.exp( -(lindex) ** 2 / (2.0 * sigma ** 2) )
    psfgrism /= psfgrism.sum()    # normalization in order to ensure flux conservation
    psfgrism_sh = fftshift(psfgrism)
#   SPECTRAL CONVOLUTION
    cube2 = np.zeros((cube.shape[0]+ 2 * ls, cube.shape[1], cube.shape[2]))
    cube2[ls:cube.shape[0]+ls, :, :] = cube
#    specconv = ifft(fft(cube2, axis=0) * fft(psfgrism_sh.reshape(psfgrism_sh.shape[0], 1, 1), axis=0), axis=0)

    specconv = irfft(rfft(cube2, axis=0) * rfft(psfgrism_sh.reshape(psfgrism_sh.shape[0], 1, 1), axis=0), axis=0)
    
    specconv = specconv.real
    specconv = specconv[ls:cube.shape[0] + ls,:,:]
    
    return specconv


def where_slit(size, cdeltxy, slit_pa, slit_x, slit_y, slit_width):
    """Function that computes the indices where the slit is located

    Parameters
    ----------
    size: np.array
        size of the cube in the spatial directions
    cdeltxy: flt
        spatial sampling (arcsec) of the high resolution datacube
    slit_pa: flt
        position angle of the slit (deg)
    slit_x: flt
        abscissa of the center of the slit (arcsec) with respect to the center of the cube in the direction corresponding to dispersion
    slit_y: flt
        ordinates of the center of the slit (arcsec) with respect to the center of the cube in the direction orthogonal to dispersion
    slit_width: flt
        width of the slit in the spectral dispersion direction (arcsec)

    """
    
    y, x = np.indices((size[0], size[1]))
    y = (y - size[0] / 2.) * cdeltxy - slit_y
    dx = - y * np.tan(np.radians(slit_pa))
    x = (x - size[1] / 2.) * cdeltxy - slit_x - dx

    ind_out = np.where(np.abs(x) > (slit_width / 2.))
    ind_in = np.where(np.abs(x) <= (slit_width / 2.))
    
    return [ind_in, ind_out]


def lambda_shift(size, cdeltxy, cdeltl, pix, dl_pix, slit_x, direction):
    """Function that computes the wavelength shift due to the grating along the dispersion direction

    Parameters
    ----------
    size: int
        size of the cube in the spatial direction corresponding to dispersion
    cdeltxy: flt
        spatial sampling (arcsec) of the high resolution datacube
    cdeltl: flt
        spectral sampling (nm) of the high resolution datacube
    pix: flt
        spatial sampling (arcsec) of the PV map
    dl_pix: flt
        spectral sampling (nm) of the PV map
    slit_width: flt
        width of the slit in the spectral dispersion direction (arcsec)
    slit_x: flt
        abscissa of the center of the slit (arcsec) with respect to the center of the cube in the direction corresponding to dispersion
    direction: int
        spectral dispersion direction: +1 (positive) lambda grows from East to West (left to right); -1 (negative) lambda grows from West to East (right to left).

    """
    #x = (np.indices((size,)) - size / 2.) * cdeltxy - slit_x  # index in terms of arcseconds centered on the center of the slit
    x = (np.indices((size,)) - size / 2.) * cdeltxy  # index in terms of arcseconds centered on the center of the cube
    dz = direction * (x / pix) * dl_pix / cdeltl  # shift in spectral pixels of the high resolution cube in the spatial direction across the slit
    return dz


def create_cube(b0, rd, rtrunc, vd, rt, vs, sig0, slope, xcen, ycen, pa, incl, fwhm, lbda, lrange= 16.8, rlast=2.05, res=0.05125, lres=0.15, kernel=3., l0=7400., rc= 'exp'):
    """Function that computes a data cube from a exponential light distribution, rotation curve and a linear velocity dispersion

    Parameters
    ----------
    b0: flt
        central surface brightness (erg/s/arcsec2)
    rd: flt
        disk scalelength (arcsec)
    rtrunc: flt
        truncation radius (arcsec) after which the flux is null
    vd: flt
        Maximum rotation velocity (km/s)
    rt: flt
        radius at wich the maximum velocity is reached (arcsec)
    vs: flt
        systemic velocity (km/s)
    sig0: flt
        velocity dispersion (km/s)
    slope: flt
        slope of the velocity dispersion profile (km/s / arcsec)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordinates of the center (arcsec)
    pa: flt
        position angle of the major axis (degree)
    incl: flt
        inclination of the disk (degree)
    fwhm: flt
        full width at half maximum of the Gaussian function which represent the PSF (arcsec)
    lbda:flt
        central wavelength of model spectral range (A)
    lrange:flt
        spectral range in A
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec)
    lres: flt
        spectral resolution of the high resolution data (A)
    kernel: flt
        FWHM in Angstrom of the spectral PSF (assumed to be Gaussian)
    l0: flt
        wavelength of the first spectral pixel (A)
    rc: ['exp'| ' flat'| 'arctan']
        rotation curve to use in the model
        - 'exp':  exponential disk
        - 'flat': flat model
        - ' arctan': arctangent function
    plot: bool
        keyword to show a figure of the result

    """

    b = exponential_disk_intensity_2D(b0, rd, rtrunc, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    if rc=='exp':
        v = exponential_disk_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    if rc=='flat':
        v = flat_model_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    if rc=='arctan':
        v = arctangent_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)

    sig = linear_velocitydispersion_2D(sig0, slope, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    sigl = sig * lbda / (cst.c * 1e-3)

    lsize = np.int(np.round( lrange / lres))        # number of spectral sampling
#    lindex=(np.arange(lsize) - lsize / 2) * lres    #spectral sampling elements
    lindex = np.arange(lsize) * lres + l0    #spectral sampling elements

    vind = (lindex - lbda) / lbda * cst.c *1e-3              #velocity sampling elements
    
    cube = (b / sigl) / np.sqrt(2 * pi) * np.exp(-np.subtract(vind.reshape(lsize, 1, 1), v.reshape(1, b.shape[0], b.shape[1])) ** 2 / (2 * sig ** 2)) * lres

    t1 = time.time()
#    SPATIAL RESOLUTION: spatial convolution with the PSF Gaussian function
    cube = psf_gaussian_convolution_2D(cube,fwhm, xcen, ycen, rlast=rlast, res=res)
    
#    SPECTRAL RESOLUTION: spectral convolution with a Gaussian rather than a complicated grating function
    cube = spectral_convolution(cube, kernel, lbda, lrange=lrange, lres=lres)
    
    return cube


def create_cube_doublet(b0, rd, rtrunc, vd, rt, vs, sig0, slope, xcen, ycen, pa, incl, fwhm, lbda, lrange= 16.8, rlast=2.05, res=0.05125, lres=0.15, kernel=3., l0=7400.,lbdaOII= 3727.425,lbda_dist=1.395, ratio=0.8, rc= 'exp'):
    """Function that computes a data cube from a exponential light distribution, rotation curve and a linear velocity dispersion
        
    Parameters
    ----------
    b0: flt
        central surface brightness (erg/s/arcsec2)
    rd: flt
        disk scalelength (arcsec)
    rtrunc: flt
        truncation radius (arcsec) after which the flux is null
    vd: flt
        Maximum rotation velocity (km/s)
    rt: flt
        radius at wich the maximum velocity is reached (arcsec)
    vs: flt
        systemic velocity (km/s)
    sig0: flt
        velocity dispersion (km/s)
    slope: flt
        slope of the velocity dispersion profile (km/s / arcsec)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordinates of the center (arcsec)
    pa: flt
        position angle of the major axis (degree)
    incl: flt
        inclination of the disk (degree)
    fwhm: flt
        full width at half maximum of the Gaussian function which represent the PSF (arcsec)
    lbda:flt
        central wavelength of model spectral range (A)
    lrange:flt
        spectral range in A
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec)
    lres: flt
        spectral resolution of the high resolution data (A)
    kernel: flt
        FWHM in Angstrom of the spectral PSF (assumed to be Gaussian)
    l0: flt
        wavelength of the first spectral pixel (A)
    lbdaOII: flt
        rest-frame mean wavelength of [OII] doublet, 3728.82 Angs + 3726.03 Angs (A)
    lbda_dist: flt
        rest-frame mean distance of the 2 lines of [OII] doublet from their mean lambda (A)
    ratio: flt
        ratio between the intensity of the 2 line of the doublet
    rc: ['exp'| ' flat'| 'arctan']
        rotation curve to use in the model
        - 'exp':  exponential disk
        - 'flat': flat model
        - ' arctan': arctangent function
    plot: bool
        keyword to show a figure of the result
        
        """
    
    b = exponential_disk_intensity_2D(b0, rd, rtrunc, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    if rc=='exp':
        v = exponential_disk_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    if rc=='flat':
        v = flat_model_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    if rc=='arctan':
        v = arctangent_velocity_2D(vd, rt, vs, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)

    sig = linear_velocitydispersion_2D(sig0, slope, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    sigl = sig * lbda / (cst.c * 1e-3)
    
    lsize = np.int(np.round( lrange / lres))        # number of spectral sampling
    #    lindex=(np.arange(lsize) - lsize / 2) * lres    #spectral sampling elements
    lindex = np.arange(lsize) * lres + l0    #spectral sampling elements
    
    v_dist= lbda_dist / lbdaOII * cst.c *1e-3   #km/s distance (velocity) of the 2 [OII] lines from their mean lambda
    vind1 = ((lindex - lbda) / lbda * cst.c *1e-3) - v_dist              #velocity sampling elements
    vind2 = ((lindex - lbda) / lbda * cst.c *1e-3) + v_dist
    
    cube1 = ratio * (b / sigl) / np.sqrt(2 * pi) * np.exp(-np.subtract(vind1.reshape(lsize, 1, 1), v.reshape(1, b.shape[0], b.shape[1])) ** 2 / (2 * sig ** 2)) * lres
    cube2 = (b / sigl) / np.sqrt(2 * pi) * np.exp(-np.subtract(vind2.reshape(lsize, 1, 1), v.reshape(1, b.shape[0], b.shape[1])) ** 2 / (2 * sig ** 2)) * lres
    cube = cube1 + cube2
    
    t1 = time.time()
    #    SPATIAL RESOLUTION: spatial convolution with the PSF Gaussian function
    cube = psf_gaussian_convolution_2D(cube,fwhm, xcen, ycen, rlast=rlast, res=res)
    
    #    SPECTRAL RESOLUTION: spectral convolution with a Gaussian rather than a complicated grating function
    cube = spectral_convolution(cube, kernel, lbda, lrange=lrange, lres=lres)
    
    return cube


def instrument(cube, cdeltxy, cdeltl, pix, dl_pix, direction=-1, slit_pa=0., slit_x=0., slit_y=0., slit_width=1.):
    """Function that creates the observed PV-map from a high resolution datacube
    We make the assumption that the dispersion will be done across the X axis and that Y axis corresponds to the North. At the end, North is up.

    0,0 corresponds to the center of the spatial dimensions of the cube

    Parameters
    ----------
    cube: numpy 3D array
        high resolution datacube after spatial smoothing
    cdeltxy: flt
        spatial sampling (arcsec) of the high resolution datacube
    cdeltl: flt
        spectral sampling (nm) of the high resolution datacube
    pix: flt
        spatial sampling (arcsec) of the PV map
    dl_pix: flt
        spectral sampling (nm) of the PV map.
    direction: int
        spectral dispersion direction: +1 (positive) lambda grows from East to West (left to right); -1 (negative) lambda grows from West to East (right to left).
    slit_pa: flt
        position angle of the slit (deg) measured from the North (Y axis, up) to the East (X axis, left)
    slit_x: flt
        abscissa of the center of the slit (arcsec) with respect to the center of the cube
    slit_y: flt
        ordinates of the center of the slit (arcsec) with respect to the center of the cube
    slit_width: flt
        width of the slit in the spectral dispersion direction, i.e. X axis (arcsec)
    """

    direction = np.sign(direction)
    #We compute the offsets due to the dispersion
    dz = lambda_shift(np.shape(cube)[2], cdeltxy, cdeltl, pix, dl_pix, slit_x, direction)
    dzz = np.int16(np.round(dz[0]))

    #We keep only information on the slit
    [ind_in, ind_out] = where_slit((np.shape(cube)[1], np.shape(cube)[2]), cdeltxy, slit_pa, slit_x, slit_y, slit_width)

    #We shift spectrally each line in the slit accross the dispersion direction (this is the time consuming part)
    cube_shift = cube * 0
    for i in np.arange(np.min(ind_in[1]), np.max(ind_in[1]) + 1):
        if dzz[i] > 0:
            cube_shift[:, :, i] = np.pad(cube[:, :, i], ((dzz[i], 0), (0, 0)), mode='edge')[:-dzz[i], :]
        else:
            cube_shift[:, :, i] = np.pad(cube[:, :, i], ((0, -dzz[i]), (0, 0)), mode='edge')[-dzz[i]:, :]

    cube_shift[:, ind_out[0], ind_out[1]] = 0.
    
    #We compute the unbinned PV map - sum accross x axis
    pvmap = cube_shift.sum(axis=2)

    #We rebin spatially and spectrally
    spectral_bin = dl_pix / cdeltl
    spatial_bin = pix / cdeltxy
    pvmap_view = pvmap.reshape(pvmap.shape[0] / spectral_bin, spectral_bin, pvmap.shape[1] / spatial_bin, spatial_bin)
    pvmap_final = pvmap_view.sum(axis=(1, 3))

    #return pvmap_final.transpose()

    size = pvmap_final.shape[1]
    y = (np.indices((size,)) - size / 2.) * pix - slit_y
    dx = -y * np.tan(np.radians(slit_pa))
    dz = -direction * dx / pix
    #dzz = np.int16(np.round(dz[0]))
    pvmap_untilted = pvmap_final.copy()
    index = np.arange(pvmap_final.shape[0])
    for i in range(pvmap_final.shape[1]):
#        print(i)
        f = interp1d(index, pvmap_final[:, i], kind='cubic', bounds_error=False, fill_value=0)
        pvmap_untilted[:, i] = f(index - dz[0][i])
        #if dzz[i] > 0:
            #pvmap_untilted[:, i] = np.pad(pvmap_final[:, i], (np.int(dzz[i]), 0), mode='edge')[:-dzz[i]]
        #else:
            #pvmap_untilted[:, i] = np.pad(pvmap_final[:, i], (0, -np.int(dzz[i])), mode='edge')[-dzz[i]:]

    return pvmap_untilted.transpose()


def pv_plot():
    plt.figure()
    plt.subplot(1, 2, 1)       # 1 row, 2 columns, plot n.1
    plt.imshow(cube[:, :, xcen / cdeltxy].transpose(), origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.subplot(1, 2, 2)       # 1 row, 2 columns, plot n.2
    plt.imshow(pvmap, origin='lower', interpolation='nearest')
    plt.colorbar()
    plt.show()
    return

def create_fitsfile(data, new_file='new'):
    """
        """
    hdr=pf.getheader('MOS_CosmosHR_P01_Exp1_MOS_SCIENCE_FLUX_EXTRACTED.fits')
    hdu=pf.PrimaryHDU(data=data, header=hdr)
    hdulist=pf.HDUList([hdu])
    hdulist.writeto("%s.fits" % (new_file), checksum=True)
    return


if __name__ == "__main__":
    
    lbda = 7400.  # ang
    direction = -1  # direction of dispersion (from West to East)

    #VIMOS data resolution
    pix = 0.205  # arcsec (pixel scale)
    dl_pix = 0.582  # A/pixel (grism dispersion)
    R = 2500.
    slit_width = 1.  # arcsec
    #kernel = lbda / R  # Angstrom (grism resolution in Angstrom for a 1" slit, R=lambda(undeviated)/deltalambda=2500, since lambda(undeviated)=7400A, then deltalamba=3A
    kernel = .5  # Angstrom (grism resolution in Angstrom for a 1" slit, R=lambda(undeviated)/deltalambda=2500, since lambda(undeviated)=7400A, then deltalamba=3A
    #spectral_resolution_grism = 3.  # Angstrom (grism resolution in Angstrom for a 1" slit, R=lambda(undeviated)/deltalambda=2500, since lambda(undeviated)=7400A, then deltalamba=3A
    
    rlast = pix * 25  # (arcsec) (in order to have an integer number of spatial pixels)
    lrange = dl_pix * 40 # ang  (in order to have an integer number of spectral pixels)
    
    #High resolution of the model:
    fwhm = 0.6  # arcsec
    cdeltxy = pix / 4  #0.05125  # arcsec ACS: 0.05'', WFC3: 0.13''
    cdeltl = dl_pix / 4  #0.1455  # A <=> R ~ 40000
    xcen = rlast
    ycen = rlast
    pa = 0.
    incl = 45.
    b0 = 2000.
    rd = 1.  # arcsec
    rtrunc = 2. # arcsec
    vd = 100.  # km/s
    rt = 1.5  # arcsec
    vs = 0.
    sig0 = 20  # km/s
    slope = 0.
    lbdaOII= 3727.425
    lbda_dist=1.395
    ratio=0.8    #ratio peak intensity between the 2 line of the doublet
    rc='exp'    #['exp'|'flat'|'arctan']

    cube = create_cube(b0, rd, rtrunc, vd, rt, vs, sig0, slope, xcen, ycen, pa, incl, fwhm, lbda, lrange=lrange, rlast=rlast, res=cdeltxy, lres=cdeltl, kernel=kernel,l0=lbda - lrange/2, rc= rc)
#    cube = create_cube_doublet(b0, rd, rtrunc, vd, rt, vs, sig0, slope, xcen, ycen, pa, incl, fwhm, lbda, lrange=lrange, rlast=rlast, res=cdeltxy, lres=cdeltl, kernel=kernel,l0=lbda - lrange/2,lbdaOII= lbdaOII,lbda_dist=lbda_dist, ratio=ratio, rc= rc)

    pvmap = instrument(cube, cdeltxy, cdeltl, pix, dl_pix, direction=direction, slit_pa=pa, slit_x=0., slit_y=0., slit_width=slit_width)

    bkg=np.random.normal(loc=0.6,scale=5.,size=(pvmap.shape[0],pvmap.shape[1]))
#    pvmap=pvmap + bkg


    pv_plot()
    hdu=pf.PrimaryHDU(data=pvmap)
    hdu.header.append(('CRVAL1', lbda - lrange/2, 'Wavelength of reference pixel'))
    hdu.header.append(('CRPIX1', 1., 'Reference pixel in X'))
    hdu.header.append(('CTYPE1', 'LINEAR','pixel coordinate system'))
    hdu.header.append(('CRVAL2', 1., 'Reference pixel'))
    hdu.header.append(('CRPIX2', 1., 'Reference pixel in Y'))
    hdu.header.append(('CTYPE2', 'PIXEL','pixel coordinate system'))
    hdu.header.append(('CD1_1', dl_pix, 'Pixel size in Angstroms'))
    hdu.header.append(('CDELT1', dl_pix, 'Pixel size in Angstroms'))
    hdu.header.append(('CD1_2', 0., 'Pixel size in Angstroms'))
    hdu.header.append(('CD2_1', 0., 'Pixels'))
    hdu.header.append(('CD2_2', 1., 'Pixels'))
    hdu.header.append(('HIERARCH ESO INS PIXSCALE', pix, 'Arcsec'))
    hdu.header.append(('HIERARCH ESO GRISM RES', kernel, 'Grism spectra resolution for a 1" slit in Angstroms'))
    hdu.header.append(('HIERARCH ESO SLIT WIDTH', slit_width, 'Slit width in arcsec'))
    hdu.header.append(('HIERARCH ESO COMPUTED SEEING', fwhm, 'Seeing computed from star spectra in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL RLAST', rlast, 'Half size of the data in spatial direction in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL LRANGE', lrange, 'Spectral range in Angstroms'))
    hdu.header.append(('HIERARCH ESO MODEL CDELTXY', cdeltxy, 'Spatial resolution of high resolution model in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL B0', b0, 'Central surface brightness'))
    hdu.header.append(('HIERARCH ESO MODEL RD', rd, 'Exponantial scalelength in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL RTRUNC', rtrunc, 'Truncation radius in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL VD', vd, 'Maximum velocity in km/s'))
    hdu.header.append(('HIERARCH ESO MODEL RT', rt, 'Transition radius (RC) in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL VS', vs, 'Systemic velocity in km/s'))
    hdu.header.append(('HIERARCH ESO MODEL SIG0', sig0, 'Central velocity dispersion in km/s'))
    hdu.header.append(('HIERARCH ESO MODEL SLOPE', slope, 'Slope of the dispersion profile km/s/arcsec (?)'))
    hdu.header.append(('HIERARCH ESO MODEL LRES', cdeltl, 'Spectral resolution of high resolution model in Angstroms'))
    hdu.header.append(('HIERARCH ESO MODEL LAMBDA', lbda, 'Systemic wavelength in Angstroms'))
    hdu.header.append(('HIERARCH ESO MODEL CENTER', xcen, 'Center with respect to bottom in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL INC', incl, 'Inclisnation in degree'))
    hdu.header.append(('HIERARCH ESO MODEL PA', pa, 'Position angle of the slit (and of the galaxy) in degree'))
    hdulist=pf.HDUList(hdu)
#    hdulist.writeto("pv_3D_vmax{}_pa{}_transpose.fits".format(int(vd),int(pa)), checksum=True, clobber=True)
#    hdulist.writeto("pv_3D.fits", checksum=True, clobber=True)
