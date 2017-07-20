import numpy as np
from scipy.special import i0, i1, k0, k1


def exponential_velocity(r, rt, vm):
    """
    Velocity function for an exponential disk

    :param ndarray r: 2D array which contain the radius
    :param int rt: radius at which the maximum velocity is reached
    :param float vm: Maximum velocity of the model
    """

    rd = rt / 2.15        # disk scale length
    vr = np.zeros(np.shape(r))
    q = np.where(r != 0)      # To prevent any problem in the center

    vr[q] = r[q] / rd * vm / 0.88 * np.sqrt(i0(0.5 * r[q] / rd) * k0(0.5 * r[q] / rd) - i1(0.5 * r[q] / rd) * k1(0.5 * r[q] / rd))

    return vr


def flat_velocity(r, rt, vm):
    """
    Velocity function for flat disk

    :param ndarray r: 2D array which contain the radius
    :param int rt: radius at which the maximum velocity is reached
    :param float vm: maximum velocity of the model
    """

    vr = np.zeros(np.shape(r))

    vr[np.where(r <= rt)] = vm*r[np.where(r <= rt)]/rt
    vr[np.where(r > rt)] = vm

    return vr


def arctan_velocity(r, rt, vm):
    """

    :param ndarray r:
    :param int rt:
    :param float vm:
    :return:
    """

    return 2*vm/np.pi*np.arctan(2*r/rt)

# Must be at th end of the file
list_model = {'exp': exponential_velocity, 'flat': flat_velocity, 'arctan': arctan_velocity}
