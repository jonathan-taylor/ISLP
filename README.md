# ISLP

This package collects data sets and various helper functions
for ISLP.

## Install instructions

We generally recommend creating a [conda](https://anaconda.org) environment to isolate any code
from other dependencies. The `ISLP` package does not have unusual dependencies, but this is still
good practice. To create a conda environment in a Mac OS X or Linux environment run:

```{python}
conda create --name islp
```

To run python code in this environment, you must activate it:

```{python}
conda activate islp
```

### Mac OS X

Running in the desired environment, we use `pip` to install the `ISLP` package:

```{python}
pip install ISLP
```

### Windows

See the [python packaging instructions](https://packaging.python.org/en/latest/tutorials/installing-packages/#ensure-you-can-run-pip-from-the-command-line) for a simple way to run `pip` within
Jupyter or IPython.

Alternatively, within a python shell in the environment you want to install `ISLP`, the following commands should install `ISLP`:

```{python}
import os, sys
cmd = f'{sys.executable} -m pip install ISLP'
os.system(cmd)
```

### Torch requirements

The `ISLP` labs use `torch` and various related packages for the lab on deep learning. The requirements
can be found [here](torch_requirements.txt). Alternatively, you can install them directly using `pip`

```{python}
!pip install -r https://raw.githubusercontent.com/intro-stat-learning/ISLP/master/torch_requirements.txt
```

## Documentation

See the [read the docs page](https://islp.readthedocs.io/en/latest/models.html) for the latest documentation.



