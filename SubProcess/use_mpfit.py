import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
import time
from Class.Model2D import Model2D
import Tools.cap_mpfit as mpfit


def use_mpfit(model, params, quiet=False):
    """

    :param Model2D model:
    :param dict params:
    :param Bool quiet:
    """

    if quiet is True:
        verbose = 0
    else:
        verbose = 1

    # PARINFOS
    p0 = [params[key] for key in params['parname']]
    parinfo = [{'value': 0., 'fixed': 0, 'limited': [0, 0], 'limits': [0., 0.], 'parname': 0., 'step': 0.} for i in range(len(p0))]

    for i in range(len(p0)):
        parinfo[i]['value'] = p0[i]['value']
        parinfo[i]['parname'] = params['parname'][i]
        parinfo[i]['limited'] = [1, 1]
        parinfo[i]['limits'] = params[params['parname'][i]]['limits']
        if params[params['parname'][i]]['fixed'] is None:
            params[params['parname'][i]]['fixed'] = 0
        if params[params['parname'][i]]['fixed'] == 1:
            parinfo[i]['fixed'] = 1

    print('\n start fit with mpfit')
    t1 = time.time()

    model_fit = mpfit.mpfit(model.least_square, parinfo=parinfo, autoderivative=1, gtol=1e-10, ftol=1e-10, xtol=1e-10, quiet=verbose)

    t2 = time.time()
    print(' fit done in: {:6.2f} s\n'.format(t2-t1))

    # Print results on screen
    print(' fit status:', model_fit.status)
    print(' Chi2R: {} DOF: {}'.format(model_fit.fnorm/model_fit.dof, model_fit.dof))

    print('', '-' * (len(params['parname'])*12))
    print('{0:^{width}}{1:^{width}}{2:^{width}}{3:^{width}}{4:^{width}}'
          '{5:^{width}}{6:^{width}}'.format(*params['parname'], width=12))
    print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}{4:^{width}.{prec}f}'
          '{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*model_fit.params, width=12, prec=6))
    print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}{4:^{width}.{prec}f}'
          '{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*model_fit.perror, width=12, prec=6))
    print('', '-' * (len(params['parname'])*12))

    return {'results': {params['parname'][i]: {'value': model_fit.params[i], 'error': model_fit.perror[i]} for i in range(len(params['parname']))},
            'mpfit': {'chi2r': model_fit.fnorm/model_fit.dof, 'dof': model_fit.dof}}
