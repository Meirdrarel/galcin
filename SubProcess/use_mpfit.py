import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
import time
from Class.Model2D import Model2D
import Tools.cap_mpfit as mpfit
import logging

logger = logging.getLogger('__galcin__')


def use_mpfit(model, params, confmeth, quiet=False):
    """
        Function call MPFIt to perform the fit of the model

    :param Model2D model: model initialised
    :param dict params: dictionary which contain all parameters with the limits and if are fixed or not
    :param dict confmeth: dictionary with method parameters
    :param Bool quiet: print or not verbose from the fit method
    """
    if confmeth['ftol'] and confmeth['ftol'] is None:
        ftol = 1e-10
    else:
        ftol = confmeth['ftol']

    if confmeth['gtol'] and confmeth['gtol'] is None:
        gtol = 1e-10
    else:
        gtol = confmeth['gtol']

    if confmeth['xtol'] and confmeth['xtol'] is None:
        xtol = 1e-10
    else:
        xtol = confmeth['xtol']

    if quiet is True:
        verbose = 0
    else:
        verbose = 1

    # create the array of dictionary 'parinfo' needed by mpfit
    p0 = [params[key] for key in params['parname']]
    parinfo = [{'value': 0., 'fixed': 0, 'limited': [1, 1], 'limits': [0., 0.], 'parname': 0., 'step': 0.} for i in range(len(p0))]

    for i in range(len(p0)):
        parinfo[i]['value'] = p0[i]['value']
        parinfo[i]['parname'] = params['parname'][i]
        parinfo[i]['limits'] = params[params['parname'][i]]['limits']
        if params[params['parname'][i]]['fixed'] is None:
            params[params['parname'][i]]['fixed'] = 0
        if params[params['parname'][i]]['fixed'] == 1:
            parinfo[i]['fixed'] = 1

    logger.info(' start fit with mpfit')
    t1 = time.time()

    # ###  Call mpfit
    model_fit = mpfit.mpfit(model.least_square, parinfo=parinfo, autoderivative=1, gtol=gtol, ftol=ftol, xtol=xtol, quiet=verbose)
    # ###

    t2 = time.time()
    logger.info(' fit done in: {:6.2f} s\n'.format(t2-t1))

    # Print results on the prompt screen
    logger.info(' fit status: {}'.format(model_fit.status))
    logger.info(' Chi2R: {} DOF: {}'.format(model_fit.fnorm/model_fit.dof, model_fit.dof))

    logger.info(' {0:^{width}}{1:^{width}}{2:^{width}}{3:^{width}}{4:^{width}}'
                '{5:^{width}}{6:^{width}}'.format(*params['parname'], width=12))
    logger.info('-' * (len(params['parname'])*12))
    logger.info(' {0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}{4:^{width}.{prec}f}'
                '{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*model_fit.params, width=12, prec=6))
    logger.info(' {0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}{4:^{width}.{prec}f}'
                '{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*model_fit.perror, width=12, prec=6))

    return {'results': {params['parname'][i]: {'value': model_fit.params[i], 'error': model_fit.perror[i]} for i in range(len(params['parname']))},
            'mpfit': {'chi2r': model_fit.fnorm/model_fit.dof, 'dof': model_fit.dof}}
