# ISLP

This package collects data sets and various helper functions
for ISLP.

## Install instructions

### Mac OS X

```{python}
pip install ISLP
```

### Windows

See the [python-packaging-instructions](https://packaging.python.org/en/latest/tutorials/installing-packages/#ensure-you-can-run-pip-from-the-command-line) for a simple way to run `pip` within
Jupyter.

Alternatively, within a python shell, the following commands should install `ISLP`:

```{python}
import os, sys
cmd = f'{sys.executable} -m pip install ISLP'
os.system(cmd)
```

### Torch requirements

The `ISLP` labs use `torch` and various related packages for the lab on deep learning. The requirements
can be found [here](requirements.txt). Alternatively, you can install them directly using `pip`

```{python}
reqs = 'https://raw.githubusercontent.com/jonathan-taylor/ISLP/master/requirements.txt'
cmd = f'{sys.executable} -m pip install -r {reqs}'
os.system(cmd)
```

## Documentation

See the [read the docs](https://islp.readthedocs.io/en/latest/models.html)



