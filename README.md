# Subset Extended Kalman Filter

Updating Neural Network models as changes in the underlying system occur:

- slowly and monotonically (drift)
- discretely (transfer learning)

## Installation

The only required dependencies aside from the standard library are `numpy>=1.8.0` (for `.npz` file saving) and `torch>=2.0.0` (for `torch.func` functions)

The `sekf` package can be installed using `pip`:

- `pip install sekf` to install from pypi
- `pip install git+https://github.com/joshuaeh/Subset-Extended-Kalman-Filter.git` to install from the github repository
- clone the repository using `git clone https://github.com/joshuaeh/Subset-Extended-Kalman-Filter.git`, then install locally using `pip install -e <path_to_repo>`
- If you are getting a `ModuleNotFoundError`, check that you are using the correct python environment. I've also found it helpful to declare the package name when installing locally using `uv`: `uv pip install -e "sekf @ ."`.  

## TODO

- [ ] documentation
