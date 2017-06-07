# Project under development

(name) fit velocity model from velocity field of galaxy seen with MUSE. 

## Model

For compute the model, we use the the method from :
<a href="Epinat, B., Amram, P., Balkowski, C., & Marcelin, M. 2010, MNRAS, 401, 2113">
http://adsabs.harvard.edu/abs/2010MNRAS.401.2113E

## There is two methods implemented

- MPFIT:
 Which uses the Levenberg-Marquardt technique to solve the
 least-squares problem.  In its typical use, MPFIT will be used to
 fit a user-supplied function (the "model") to user-supplied data
 points (the "data") by adjusting a set of parameters.  MPFIT is
 based upon MINPACK-1 (LMDIF.F) by More' and collaborators.

- Pymultinest : use a Nested Sampling Monte Carlo library, for more information about it see:
                https://johannesbuchner.github.io/PyMultiNest/



## Compatible with MPI4PY

For more information and how to install it, refer to its site :
http://pythonhosted.org/mpi4py/

To execute the program with mpi4py:

    mpiexec -n (nb core) main_mpi.py **args

more information ASAP, the final name and the repository will be changed at the end of the development.
