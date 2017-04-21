import math
import numpy as np
import tools
import velocity_model as vm
from PSF import PSF
from Images import Image


class Model2D:

    def __init__(self, flux_ld, flux_hd, sig0, slope=0.):
        """

        :param float sig0: velocity dispersion of the model
        :param float slope: slope of the velocity dispersion
        :param Image flux_ld: image of observed galaxy
        :param Image flux_hd: image of observed galaxy
        """

        self.model_pos_angl = None
        self.model_incl = None
        self.model_xcen = None
        self.model_ycen = None
        self.model_vmax = None
        self.model_syst_vel = None
        self.model_sig0 = sig0
        self.model_slope = slope
        self.model_charac_rad = None
        self.model_radius = None
        self.model_theta = None
        self.model_size = None
        self.vel_map = np.zeros(flux_ld.size)
        self.vel_map_hd = np.zeros(flux_hd.size)
        self.vel_masked = None

    def set_parameters(self, xcen, ycen, pos_angl, incl, syst_vel, max_vel, charac_rad, flux_hd):
        """

        :param float xcen: abscissa of the center in pixel
        :param float ycen: ordinate of the center in pixel
        :param float pos_angl: position angle of the major axis in degree
        :param float incl: inclination of the disk in degree
        :param float syst_vel: systemic velocity in km/s
        :param float max_vel: maximum rotation velocity in km/s
        :param float charac_rad: characteristic radius, where the maximum velocity is reached
        :param Image flux_hd:
        """

        self.model_pos_angl = pos_angl
        self.model_incl = incl
        self.model_xcen = xcen * flux_hd.oversample
        self.model_ycen = ycen * flux_hd.oversample
        self.model_vmax = max_vel
        self.model_syst_vel = syst_vel
        self.model_charac_rad = charac_rad*flux_hd.oversample
        self.model_size = np.array(flux_hd.size)
        self.model_radius, self.model_theta = tools.sky_coord_to_galactic(self.model_xcen, self.model_ycen, self.model_pos_angl, self.model_incl,
                                                                          im_size=self.model_size)

    def disk_velocity(self, vel_model):
        """

        :param vel_model:
        :return ndarray:
        """

        vr = vel_model(self.model_radius, self.model_charac_rad, self.model_vmax)

        # Calculation of the velocity field
        v = vr * math.sin(math.radians(self.model_incl)) * self.model_theta + self.model_syst_vel

        return v

    def velocity_map(self, psf, flux_ld, flux_hd, vel_model):
        """

        :param PSF psf:
        :param Image flux_ld:
        :param Image flux_hd:
        :param function vel_model:
        """

        self.vel_map_hd = self.disk_velocity(vel_model)

        vel_times_flux = tools.rebin_data(psf.convolution(flux_hd.data * self.vel_map_hd), flux_hd.oversample)

        self.vel_map[flux_ld.mask] = vel_times_flux[flux_ld.mask] / flux_ld.data[flux_ld.mask]

    def linear_velocity_dispersion(self):
        """

        :return ndarray:
        """
        # Calculation of the velocity dispersion
        sig = self.model_sig0 + self.model_slope * np.abs(self.model_radius)
        sig[np.where(sig <= 0)] = 0

        return sig

    def square_vel_disp(self, flux_ld, flux_hd, psf):
        """

        :param PSF psf:
        :param Image flux_ld:
        :param Image flux_hd:
        :return ndarray:
        """

        term1 = np.zeros(flux_ld.size)
        term2 = np.zeros(flux_ld.size)
        term3 = np.zeros(flux_ld.size)

        sig = self.linear_velocity_dispersion()

        term1[flux_ld.mask] = tools.rebin_data(psf.convolution(sig ** 2 * flux_hd.data), flux_hd.oversample)[flux_ld.mask] / flux_ld.data[flux_ld.mask]
        term2[flux_ld.mask] = tools.rebin_data(psf.convolution(self.vel_map_hd ** 2 * flux_hd.data), flux_hd.oversample)[flux_ld.mask] / flux_ld.data[
            flux_ld.mask]
        term3[flux_ld.mask] = (tools.rebin_data(psf.convolution(self.vel_map_hd * flux_hd.data), flux_hd.oversample)[flux_ld.mask] /
                               flux_ld.data[flux_ld.mask]) ** 2

        return term1 + term2 - term3

    def get_parameter(self, flux_hd):
        """
        
        :param Image flux_hd:
        """
        return [self.model_xcen/flux_hd.oversample, self.model_ycen/flux_hd.oversample, self.model_pos_angl, self.model_incl, self.model_syst_vel,
                self.model_vmax, self.model_charac_rad/flux_hd.oversample]

