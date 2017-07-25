# name

### What is it?
(name) fit velocity model from velocity field of galaxy seen with MUSE. 


### Model

For compute the model, we use method of moments from :
<a href="http://adsabs.harvard.edu/abs/2010MNRAS.401.2113E">
Epinat, B., Amram, P., Balkowski, C., & Marcelin, M. 2010, MNRAS, 401, 2113</a>

But you can add more rotational curves in "velocity_model.py" or create another model like cube etc.

### There is two methods implemented

- MPFIT: Which uses the Levenberg-Marquardt technique to solve the
 least-squares problem.

- PyMultiNest : use a Nested Sampling Monte Carlo library and bayesian statistics.

## Installation

The program need PyMultiNest installed, please refer<a href="https://johannesbuchner.github.io/PyMultiNest/"> here</a>for 
its installation guide.

The program need also **astropy** and **yaml** libraries.

## How to lunch?
(name) can be lunch from prompt with
```
main.py path filename
```
or in an python script/console by importing first and call the main function
```
import main.py
    
main.main(path, filename, rank=0)
```
The program need a config file in **YAML**.

### Compatible with MPI4PY

For more information and how to install it, follow<a href="http://pythonhosted.org/mpi4py/"> this link<a/>.
To execute the program with **mpi4py**:
```
mpiexec -n (nbcore) main.py path filename
```
## Output

Outputs are written in a directory where the **YAML** config file is. The name of the directory depend of the method, the model and fixed paramters like: 
method_model_fixedparams. A recapitulation of parameters of the model is written in the header of fits file and in a **YAML** file. 



