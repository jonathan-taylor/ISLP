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

On windows, create a `Python` environment called `islp` in the Anaconda app. This can be done by selecting `Environments` on the left hand side of the app's screen. After creating the environment, open a terminal within that environment by clicking on the <font color="green">Play</font> button.


## Installing `ISLP`

Running in the desired environment, we use `pip` to install the `ISLP` package:

```{python}
pip install ISLP
```

### Torch requirements

The `ISLP` labs use `torch` and various related packages for the lab on deep learning. The requirements
can be found [here](torch_requirements.txt). Alternatively, you can install them directly using `pip` within a terminal. As above, ensure
that you have activated that conda environment (Mac OS or Linux) or
started a terminal within that environment from the Anaconda app (Windows).

```{python}
pip install -r https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/torch_requirements.txt
```

## Documentation

See the [read the docs page](https://islp.readthedocs.io/en/latest/models.html) for the latest documentation.



