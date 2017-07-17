import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))

import pymultinest
import time
from Class.Model2D import Model2D
import matplotlib.pyplot as plt


def use_pymultinest(model, params, quiet=True, nbp=0, pltstats=False, rank=0, path=None, whd=None):
    """

    :param Model2D model:
    :param dict params:
    :param int quiet: 
    :param string whd:
    :param int nbp:
    """

    def prior(cube, ndim, nparams):
        """
        define the limits where multinest have to explore

        :param ndarray cube: data whith n_params dimension
        :param int ndim: number of dimension if different of the number of paramerters
        :param int nparams: number of parameters
        """
        if nparams == 7:
            for i in range(nparams):
                limits = params[params['parname'][i]]['limits']
                cube[i] = cube[i] * (limits[1] - limits[0]) + limits[0]

    if rank == 0:
        t1 = time.time()

    pymultinest.run(model.log_likelihood, prior, len(params['parname']), outputfiles_basename=path+'/res'+whd+'_', resume=False, verbose=quiet, max_iter=nbp,
                    n_live_points=50, sampling_efficiency=0.8, evidence_tolerance=0.5, n_iter_before_update=100, null_log_evidence=-1e90,
                    max_modes=100, mode_tolerance=-1e60)

    if rank == 0:
        t2 = time.time()

        print('\n fit done in: {:6.2f} s \n'.format(t2-t1))

        output = pymultinest.Analyzer(n_params=len(params['parname']), outputfiles_basename=path+'/res'+whd+'_')
        bestfit = output.get_best_fit()
        stats = output.get_mode_stats()

        print('', '-' * (len(params['parname'])*12))
        print('{0:^{width}}{1:^{width}}{2:^{width}}{3:^{width}}{4:^{width}}{5:^{width}}'
              '{6:^{width}}'.format(*params['parname'], width=12))
        print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}'
              '{4:^{width}.{prec}f}{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*bestfit['parameters'], width=12, prec=6))
        print('{0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}'
              '{4:^{width}.{prec}f}{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*stats['modes'][0]['sigma'], width=12, prec=6))
        print('', '-' * (len(params['parname'])*12))

        # plot all parameters probabilities in a pdf file
        if pltstats is True:
            p = pymultinest.PlotMarginalModes(output)
            plt.figure(figsize=(5 * len(params['parname']), 5 * len(params['parname'])))
            for i in range(len(params['parname'])):
                plt.subplot(len(params['parname']), len(params['parname']), len(params['parname']) * i + i + 1)
                p.plot_marginal(i, with_ellipses=False, with_points=True, grid_points=50, only_interpolate=False)
                plt.ylabel("Probability")
                plt.xlabel(params['parname']()[i])

                for j in range(i):
                    plt.subplot(len(params['parname']), len(params['parname']), len(params['parname']) * j + i + 1)
                    p.plot_conditional(i, j, with_ellipses=False, with_points=True, grid_points=30, only_interpolate=False)
                    plt.xlabel(params['parname']()[i])
                    plt.ylabel(params['parname']()[j])

            plt.savefig(path+'/proba'+whd+'.pdf', bbox_inches='tight')

        return {'results': {params['parname'][i]: {'value': bestfit['parameters'][i],
                                                   'error': stats['modes'][0]['sigma'][i]} for i in range(len(params['parname']))},
                'PyMultiNest': {'log likelihood': bestfit['log_likelihood']}}

