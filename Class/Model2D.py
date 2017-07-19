import math
import numpy as np
import Tools.tools as tools
from Class.PSF import PSF
from Class.Images import Image


class Model2D:
    """
        Model in 2D of the velocity field of a galaxy
        Compute the velocity field and the dispersion of a model using observed data in input.
        This model can be used only with rotational curves with 3 parameters like which in 'velocity_model.py'

        Parameters:
            :param Image vel: represent the velocity field to fit
            :param Image errvel: the error map of the velocity (vel)
            :param Image flux: flux distribution at the same or in high resolution than the velocity
            :param PSF psf: the class of the psf
            :param func vel_model: velocity function used to create the model
            :param float sig0: velocity dispersion of the model
            :param float slope: slope of the velocity dispersion
        """
    def __init__(self, vel, errvel, flux, psf, vel_model, sig0, slope=0.):

        # None parameters
        self.model_pos_angl = None
        self.model_incl = None
        self.model_xcen = None
        self.model_ycen = None
        self.model_vmax = None
        self.model_syst_vel = None
        self.model_charac_rad = None
        self.model_radius = None
        self.model_theta = None

        # Parameters which must must be initialized
        self.vel_model = vel_model
        self.model_sig0 = sig0
        self.model_slope = slope
        self.vel_map = np.zeros(vel.get_size())
        self.vel_disp = np.zeros(vel.get_size())
        self.vel_map_hd = np.zeros(np.array(flux.get_size()))
        self.vel = vel
        self.errvel = errvel
        self.flux = flux
        self.psf = psf

    def set_parameters(self, xcen, ycen, pos_angl, incl, syst_vel, max_vel, charac_rad):
        """
            set the value of the model parameters

        :param float xcen: abscissa of the center in pixel
        :param float ycen: ordinate of the center in pixel
        :param float pos_angl: position angle of the major axis in degree
        :param float incl: inclination of the disk in degree
        :param float syst_vel: systemic velocity in km/s
        :param float max_vel: maximum rotation velocity in km/s
        :param float charac_rad: characteristic radius, where the maximum velocity is reached
        """

        self.model_pos_angl = pos_angl
        self.model_incl = incl
        self.model_xcen = (xcen + 0.5) * self.flux.get_oversamp() - 0.5
        self.model_ycen = (ycen + 0.5) * self.flux.get_oversamp() - 0.5
        self.model_vmax = max_vel
        self.model_syst_vel = syst_vel
        self.model_charac_rad = charac_rad*self.flux.get_oversamp()
        self.model_radius, self.model_theta = tools.sky_coord_to_galactic(self.model_xcen, self.model_ycen, self.model_pos_angl, self.model_incl,
                                                                          im_size=np.shape(self.vel_map_hd))

    def get_parameter(self):
        """
            Get the actual parameters of the model (in low resolution scale)

        :return ndarray:
        """
        return [(self.model_xcen + 0.5) * self.flux.get_oversamp() - 0.5, (self.model_ycen + 0.5) * self.flux.get_oversamp() - 0.5,
                self.model_pos_angl, self.model_incl, self.model_syst_vel, self.model_vmax, self.model_charac_rad/self.flux.get_oversamp()]

    def disk_velocity(self):
        """
            Compute the velocity field

        :return ndarray:
        """

        vr = self.vel_model(self.model_radius, self.model_charac_rad, self.model_vmax)

        # Calculation of the velocity field
        v = vr * math.sin(math.radians(self.model_incl)) * self.model_theta + self.model_syst_vel

        return v

    def velocity_map(self):
        """
            calculate the velocity map of the model
        """

        self.vel_map_hd = self.disk_velocity()

        vel_times_flux = tools.rebin_data(self.psf.convolution(self.flux.data * self.vel_map_hd), self.flux.get_oversamp())

        self.vel_map[self.vel.mask] = vel_times_flux[self.vel.mask] / self.flux.data_rebin[self.vel.mask]

    def linear_velocity_dispersion(self):
        """
            return the velocity dispersion map needed for the fit process

        :return ndarray: velocity dispersion map
        """
        # Calculation of the velocity dispersion
        sig = self.model_sig0 + self.model_slope * np.abs(self.model_radius)
        sig[np.where(sig <= 0)] = 0

        return sig

    def vel_disp_map(self):
        """
            calculate the velocity dispersion map from the velocity field (with bestfit parameters) and the flux distribution

        :return ndarray: velocity dispersion map
        """

        term1 = np.zeros(self.vel.size)
        term2 = np.zeros(self.vel.size)
        term3 = np.zeros(self.vel.size)

        sig = self.linear_velocity_dispersion()

        term1[self.vel.mask] = tools.rebin_data(self.psf.convolution(sig ** 2 * self.flux.data), self.flux.oversample)[self.vel.mask] \
                               / self.flux.data_rebin[self.vel.mask]
        term2[self.vel.mask] = tools.rebin_data(self.psf.convolution(self.vel_map_hd ** 2 * self.flux.data), self.flux.oversample)[self.vel.mask] \
                               / self.flux.data_rebin[self.vel.mask]
        term3[self.vel.mask] = (tools.rebin_data(self.psf.convolution(self.vel_map_hd * self.flux.data), self.flux.oversample)[self.vel.mask] /
                                self.flux.data_rebin[self.vel.mask]) ** 2

        self.vel_disp = np.sqrt(term1 + term2 - term3)

    def least_square(self, p, fjac=None):
        """
            Function minimized by mpfit.

            Return (data-model)^2/err^2

        :param ndarray p: array of parameters
        :return ndarray:
        """

        self.set_parameters(*p)
        self.velocity_map()

        return [0, np.reshape((self.vel.data[self.vel.mask]-self.vel_map[self.vel.mask])/self.errvel.data[self.vel.mask], -1)]

    def log_likelihood(self, cube, ndim, nparams):
        """
            log likelihood function which maximized by multinest

            Return sum[(data-model)^2/(2*err^2)]

        :param ndarray cube: data whith n_params dimension
        :param int ndim: number of dimension if different of the number of paramerters
        :param int nparams: number of parameters
        :return float:
        """
        self.set_parameters(cube[0], cube[1], cube[2], cube[3], cube[4], cube[5], cube[6])
        self.velocity_map()

        chi2 = -(self.vel_map[self.vel.mask] - self.vel.data[self.vel.mask])**2/(2*self.errvel.data[self.vel.mask]**2)

        return np.sum(chi2)
