# Project under development

(name) fit velocity model from velocity field of galaxy seen with MUSE. 


There is two methods implemented
-
- MPFIT:
 Which uses the Levenberg-Marquardt technique to solve the
 least-squares problem.  In its typical use, MPFIT will be used to
 fit a user-supplied function (the "model") to user-supplied data
 points (the "data") by adjusting a set of parameters.  MPFIT is
 based upon MINPACK-1 (LMDIF.F) by More' and collaborators.

- Pymultinest : use a Nested Sampling Monte Carlo library, for more information about it see:
                https://johannesbuchner.github.io/PyMultiNest/



Compatible with MPI4PY
-
For more information ans how to install it, refer to its site :
http://pythonhosted.org/mpi4py/

To execute the program with mpi4py:

    mpiexec -n (nb core) main_mpi.py

more information ASAP, the final name and the repository will be changed at the end of the development.
