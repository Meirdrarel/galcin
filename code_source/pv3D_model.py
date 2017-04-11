#!/usr/bin/env python
#@author: Debora

import ipdb
import time
import math
import numpy as np
import scipy as sp
from scipy.special import i0, i1, k0, k1
from scipy import constants as cst
import matplotlib as mpl
from matplotlib import pyplot as plt
import pyfits as pf
from numpy.fft import fft, ifft, fft2, ifft2, fftshift


def exponential_disk_intensity_2D(b0, rd, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True):
    """
    Function that computes the intensity map of an exonential disk

    Parameters
    ----------
    b0: flt
        central surface brightness (erg/s/arcsec2)
    rd: flt
        disk scalelength (arcsec)
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
    b = b0 * np.exp(- np.abs(r) / rd) * res ** 2

    # Display of the result
    if plot is True:
        plt.imshow(b, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return b


def exponential_disk_velocity_2D(vd, rt, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True):
    """Function that computes the velocity field for an exonential disk

    Parameters
    ----------
    vd: flt
        Maximum rotation velocity (km/s)
    rt: flt
        radius at wich the maximum velocity is reached (arcsec)
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
    v = vr * sin(radians(incl)) * ctheta

    # Display of the result
    if plot is True:
        plt.imshow(v, origin='lower', interpolation='nearest')
        plt.colorbar()
        plt.show()

    return v


def linear_velocitydispersion_2D(sig0, slope, xcen, ycen, pa, incl, rlast=10, res=0.04, plot=True):
    """
    Function that computes the velocity dispersion map with a linearly decreasing profile

    Parameters
    ----------
    sig0: flt
        velocity dispersion (km/s)
    slope: flt
        slope of the velocity dispersion profile (km/s / arcsec)
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
    """
    Function that computes the convolution of a datacube with a PSF 2D Gaussian function
        
    Parameters
    ----------
    cube: numpy 3D array
        high resolution datacube that has to be convolved with the PSF
    fwhm: flt
        full width at half maximum of the Gaussian function which represents the PSF (arcsec)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordiantes of the center (arcsec)
    rlast: flt
        maximum radius of the scene (arcsec), rlast should be larger than the slit length + seeing
    res: flt
        resolution of the high resolution data (arcsec), res should be at least n x pixel size       
        """

#   CREATION OF THE 2D GAUSSIAN FUNCTION
    nelement = np.int(np.ceil(2 * rlast / res))  # number of spatial sampling elements in rlast
    fwhmind=np.int(np.ceil(fwhm/res))       # number of spatial sampling elements in fwhm
    s=2*fwhmind         #(2*fwhmind) is considered to avoid the boundary effect after the convolution.
    [y, x] = np.indices((nelement+2*s, nelement+2*s)) * res  # index matrices

    sigma= fwhm / (2 * sqrt(2*log(2)))
    psf= (1./(2*pi*sigma**2)) * np.exp( -((x - xcen - s*res)**2 + (y - ycen - s*res)**2) / (2.0 * sigma ** 2) )
    psf /= psf.sum()        # normalization in order to ensure flux conservation

#   SPATIAL CONVOLUTION
    psf_sh=fftshift(psf)
    
    cube2= np.zeros((cube.shape[0],cube.shape[1]+2*s, cube.shape[2]+2*s))
    cube2[:, s:cube.shape[1]+s, s:cube.shape[2]+s]=cube

    cubeconv=ifft2(fft2(cube2) * fft2(psf_sh.reshape(1, psf.shape[0], psf.shape[1])))
    cubeconv=cubeconv.real
    cubeconv=cubeconv[:, s:cube.shape[1]+s, s:cube.shape[2]+s]

    return cubeconv

def spectral_convolution(cube, kernel, lbda, lrange=16.8, lres=0.15):
    """
    Function that computes the convolution of a datacube with a Gaussian function that represent the intrinsic spectral resolution of the GRISM (linked to the number of grooves)

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
    kernelind=np.int(np.ceil(kernel/lres))       # number of spatial sampling elements in kernel
    ls=2*kernelind                      #(2*fwhmind) is considered to avoid the boundary effect after the convolution.
    lindex=(np.arange(lsize +2*ls) - (lsize +2*ls) / 2) * lres    #spectral sampling elements
#    vind = (lindex / lbda) * cst.c *1e-3              #velocity sampling elements

    
    sigma= kernel / (2 * sqrt(2*log(2)))
    psfgrism= (1/(sigma*sqrt(2*pi))) * np.exp( -(lindex)**2 / (2.0 * sigma ** 2) )
    psfgrism /= psfgrism.sum()    # normalization in order to ensure flux conservation
    psfgrism_sh=fftshift(psfgrism)
#    ipdb.set_trace()
#   SPECTRAL CONVOLUTION
    cube2= np.zeros((cube.shape[0]+ 2*ls,cube.shape[1], cube.shape[2]))
    cube2[ls:cube.shape[0]+ls, :, :]=cube
    specconv=ifft(fft(cube2, axis=0)* fft(psfgrism_sh.reshape(psfgrism_sh.shape[0], 1, 1), axis=0), axis=0)
    specconv=specconv.real
    specconv=specconv[ls:cube.shape[0]+ls,:,:]
    
    return specconv


def where_slit(size, cdeltxy, slit_pa, slit_x, slit_y, slit_width):
    """
    Function that computes the indices where the slit is located

    Parameters
    ----------
    size: int
        size of the cube in the spatial direction orthogonal to dispersion
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
    y = (np.indices((size[0],)) - size[0] / 2.) * cdeltxy - slit_y
    dx = - y * np.tan(np.radians(slit_pa))
    x = np.indices((size[0], size[1]))[1] - size[1] / 2.
    x = x * cdeltxy - slit_x + dx.reshape((size[1], 1))

    ind_out = np.where(np.abs(x) > (slit_width / 2.))
    ind_in = np.where(np.abs(x) <= (slit_width / 2.))
    return [ind_in, ind_out]

    return ind_out


def lambda_shift(size, cdeltxy, cdeltl, pix, dl_pix, slit_x):
    """
    Function that computes the wavelength shift due to the grating along the dispersion direction

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

    """
    #x = (np.indices((size,)) - size / 2.) * cdeltxy - slit_x  # index in terms of arcseconds centered on the center of the slit
    x = (np.indices((size,)) - size / 2.) * cdeltxy  # index in terms of arcseconds centered on the center of the cube
    dz = (x / pix) * dl_pix / cdeltl  # shift in spectral pixels of the high resolution cube in the spatial direction across the slit
    return dz



def create_cube(b0, rd, vd, rt, sig0, slope, xcen, ycen, pa, incl, fwhm, lbda, lrange= 16.8, rlast=2.05, res=0.05125, lres=0.15, kernel=3., l0=7400.):
    """
    Function that computes a data cube from a exponential light distribution, rotation curve and a linear velocity dispersion

    Parameters
    ----------
    b0: flt
        central surface brightness (erg/s/arcsec2)
    rd: flt
        disk scalelength (arcsec)
    vd: flt
        Maximum rotation velocity (km/s)
    rt: flt
        radius at wich the maximum velocity is reached (arcsec)
    sig0: flt
        velocity dispersion (km/s)
    slope: flt
        slope of the velocity dispersion profile (km/s / arcsec)
    xcen: flt
        abscissa of the center (arcsec)
    ycen: flt
        ordiantes of the center (arcsec)
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
    plot: bool
        keyword to show a figure of the result

    """

    b = exponential_disk_intensity_2D(b0, rd, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    v = exponential_disk_velocity_2D(vd, rt, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)
    sig = linear_velocitydispersion_2D(sig0, slope, xcen, ycen, pa, incl, rlast=rlast, res=res, plot=False)


    lsize = np.int(np.round( lrange / lres))        # number of spectral sampling
#    lindex=(np.arange(lsize) - lsize / 2) * lres    #spectral sampling elements
    lindex=np.arange(lsize) * lres +l0    #spectral sampling elements

    vind = (lindex - lbda) / lbda * cst.c *1e-3              #velocity sampling elements


    cube = b * np.exp(-np.subtract(vind.reshape(lsize, 1, 1), v.reshape(1, b.shape[0], b.shape[1])) ** 2 / (2 * sig ** 2))

#    SPATIAL RESOLUTION: spatial convolution with the PSF Gaussian function
    cube=psf_gaussian_convolution_2D(cube,fwhm, xcen, ycen, rlast=rlast, res=res)

#    SPECTRAL RESOLUTION: spectral convolution with a Gaussian rather than a complicated grating function

    cube = spectral_convolution(cube, kernel, lbda, lrange=lrange, lres=lres)


    return cube

def instrument(cube, cdeltxy, cdeltl, pix, dl_pix, slit_pa=0., slit_x=0., slit_y=0., slit_width=1.):
    """
    Function that creates the observed PV-map from a high resolution datacube
    We make the assumption that the dispersion will be done across the X axis and that Y axis corresponds to the North.

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
        spectral sampling (nm) of the PV map
    slit_pa: flt
        position angle of the slit (deg) with respect to the North (Y axis)
    slit_x: flt
        abscissa of the center of the slit (arcsec) with respect to the center of the cube
    slit_y: flt
        ordinates of the center of the slit (arcsec) with respect to the center of the cube
    slit_width: flt
        width of the slit in the spectral dispersion direction, i.e. X axis (arcsec)
    
        """


    #We compute the offsets due to the dispersion
    dz = lambda_shift(np.shape(cube)[2], cdeltxy, cdeltl, pix, dl_pix, slit_x)
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

    #We compute the unbinned PV map
    pvmap = cube_shift.sum(axis=2)

    #We rebin spatially and spectrally
    spectral_bin = dl_pix / cdeltl
    spatial_bin = pix / cdeltxy
    pvmap_view = pvmap.reshape(pvmap.shape[0] / spectral_bin, spectral_bin, pvmap.shape[1] / spatial_bin, spatial_bin)
    pvmap_final = pvmap_view.sum(axis=(1, 3))

    size = pvmap_final.shape[1]
    y = (np.indices((size,)) - size / 2.) * pix - slit_y
    dx = -y * np.tan(np.radians(slit_pa))
    dz = dx / pix
    dzz = np.int16(np.round(dz[0]))
    pvmap_untilted = pvmap_final.copy()
    for i in range(pvmap_final.shape[1]):
        if dzz[i] > 0:
            pvmap_untilted[:, i] = np.pad(pvmap_final[:, i], (np.int(dzz[i]), 0), mode='edge')[:-dzz[i]]
        else:
            pvmap_untilted[:, i] = np.pad(pvmap_final[:, i], (0, -np.int(dzz[i])), mode='edge')[-dzz[i]:]



    return pvmap_final.transpose()
#    return pvmap_untilted.transpose()


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
    
    #VIMOS data resolution
    pix = 0.205  # arcsec (pixel scale)
    dl_pix = 0.582  # A/pixel (grism dispersion)
    spectral_resolution_grism = 3.  # Angstrom (grism resolution in Angstrom for a 1" slit, R=lambda(undeviated)/deltalambda=2500, since lambda(undeviated)=7400A, then deltalamba=3A
    slit_width = 1.  # arcsec
    #High resolution of the model:
    cdeltxy = 0.05125  # arcsec ACS: 0.05'', WFC3: 0.13''
    cdeltl = 0.1455  # A <=> R ~ 40000
    
    b0 = 10.
    rd = 1.
    rlast = pix * 14  # (arcsec) (in order to have an integer number of spatial pixels)
    vd = 200.  # km/s
    sig0 = 20  # km/s
    lres = cdeltl  # Ang
    lrange = dl_pix * 40 # A  (in order to have an integer number of spectral pixels)
    lbda = 7000.
    
    rt = 1.
    slope = 0.
    xcen = rlast
    ycen = rlast
    pa = 30.
    incl = 60.
    fwhm = 0.6
    kernel= spectral_resolution_grism

    cube = create_cube(b0, rd, vd, rt, sig0, slope, xcen, ycen, -pa, incl, fwhm, lbda, lrange=lrange, rlast=rlast, res=cdeltxy, lres=lres, kernel=kernel,l0=lbda - lrange/2)

    pvmap = instrument(cube, cdeltxy, cdeltl, pix, dl_pix, slit_pa=pa, slit_x=0.0, slit_y=0., slit_width=slit_width)

#    v=exponential_disk_velocity_2D(vd, rt, xcen, ycen, pa, incl, rlast=rlast, res=cdeltxy, plot=False)
#    intens=exponential_disk_intensity_2D(b0, rd, xcen, ycen, pa, incl, rlast=rlast, res=cdeltxy, plot=True)

#    create_fitsfile(cube, new_file='cube')
#    plt.imshow(intens, origin='lower', interpolation='nearest')
#    plt.colorbar()
#    plt.show()

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
    hdu.header.append(('HIERARCH ESO GRISM RES', spectral_resolution_grism, 'Grism spectra resolution for a 1" slit in Angstroms'))
    hdu.header.append(('HIERARCH ESO SLIT WIDTH', slit_width, 'Slit width in arcsec'))
    hdu.header.append(('HIERARCH ESO SEEING', fwhm, 'Seeing in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL RLAST', rlast, 'Half size of the data in spatial direction in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL LRANGE', lrange, 'Spectral range in Angstroms'))
    hdu.header.append(('HIERARCH ESO MODEL CDELTXY', cdeltxy, 'Spatial resolution of high resolution model in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL B0', b0, 'Central surface brightness'))
    hdu.header.append(('HIERARCH ESO MODEL RD', rd, 'Exponantial scalelength in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL VD', vd, 'Maximum velocity in km/s'))
    hdu.header.append(('HIERARCH ESO MODEL RT', rt, 'Transition radius (RC) in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL SIG0', sig0, 'Central velocity dispersion in km/s'))
    hdu.header.append(('HIERARCH ESO MODEL SLOPE', slope, 'Slope of the dispersion profile km/s/arcsec (?)'))
    hdu.header.append(('HIERARCH ESO MODEL LRES', lres, 'Spectral resolution of high resolution model in Angstroms'))
    hdu.header.append(('HIERARCH ESO MODEL LAMBDA', lbda, 'Systemic wavelength in Angstroms'))
    hdu.header.append(('HIERARCH ESO MODEL CENTER', xcen, 'Center with respect to bottom in arcsec'))
    hdu.header.append(('HIERARCH ESO MODEL INC', incl, 'Inclisnation in degree'))
    hdu.header.append(('HIERARCH ESO MODEL PA', pa, 'Position angle of the slit (and of the galaxy) in degree'))
    hdulist=pf.HDUList(hdu)
    hdulist.writeto("pv_3D_vmax{}_pa{}_transpose.fits".format(int(vd),int(pa)), checksum=True, overwrite=True)






