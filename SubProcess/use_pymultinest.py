import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Class/'), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('../Tools/'), '..')))

import pymultinest
import time
from Class.Model2D import Model2D
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger('__galcin__')


def use_pymultinest(model, params, quiet=True, nbp=0, pltstats=False, rank=0, path=None, whd=None):
    """

        Function which define the parameters's space, call PyMultiNest and perform the analysis of the results

    :param Model2D model: model initialized
    :param dict params: dictionary which contain all parameters with the limits and if are fixed or not
    :param bool quiet: print or not verbose from the fit method
    :param int nbp: number of points calculated by MultiNest, set to 0 for unlimited
    :param bool pltstats: create a pdf file with plots of probabilities of parameters
    :param int rank: the rank of the thread (needed whe you use MPI4PY with more than 1 core)
    :param str path: the relative of the directory where PyMultiNest can write files needed for the further analysis
    :param string whd: suffix for filename if a high resolution map is used
    """

    def prior(cube, ndim, nparams):
        """

            Define the limits of the parameters' space where multinest have to explore

        :param ndarray cube: data with n_params dimension
        :param int ndim: number of dimension if different of the number of parameters
        :param int nparams: number of parameters
        """
        if nparams == 7:
            for i in range(nparams):
                limits = params[params['parname'][i]]['limits']
                cube[i] = cube[i] * (limits[1] - limits[0]) + limits[0]

    if rank == 0:
        t1 = time.time()

    # ### Call PyMultiNest
    pymultinest.run(model.log_likelihood, prior, len(params['parname']), outputfiles_basename=path+'/res'+whd+'_', resume=False, verbose=quiet, max_iter=nbp,
                    n_live_points=50, sampling_efficiency=0.8, evidence_tolerance=0.5, n_iter_before_update=100, null_log_evidence=-1e90,
                    max_modes=100, mode_tolerance=-1e60)
    # ###

    if rank == 0:
        t2 = time.time()

        logger.info(' fit done in: {:6.2f} s \n'.format(t2-t1))

        output = pymultinest.Analyzer(n_params=len(params['parname']), outputfiles_basename=path+'/res'+whd+'_')
        bestfit = output.get_best_fit()
        stats = output.get_mode_stats()

        # print results ont he prompt screen
        logger.info(' {0:^{width}}{1:^{width}}{2:^{width}}{3:^{width}}{4:^{width}}{5:^{width}}'
                    '{6:^{width}}'.format(*params['parname'], width=12))
        logger.info('-' * (len(params['parname'])*12))
        logger.info(' {0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}'
                    '{4:^{width}.{prec}f}{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*bestfit['parameters'], width=12, prec=6))
        logger.info(' {0:^{width}.{prec}f}{1:^{width}.{prec}f}{2:^{width}.{prec}f}{3:^{width}.{prec}f}'
                    '{4:^{width}.{prec}f}{5:^{width}.{prec}f}{6:^{width}.{prec}f}'.format(*stats['modes'][0]['sigma'], width=12, prec=6))

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
            logger.info('Plot of probabilities saved as pdf')

        return {'results': {params['parname'][i]: {'value': bestfit['parameters'][i],
                                                   'error': stats['modes'][0]['sigma'][i]} for i in range(len(params['parname']))},
                'PyMultiNest': {'log likelihood': bestfit['log_likelihood']}}
