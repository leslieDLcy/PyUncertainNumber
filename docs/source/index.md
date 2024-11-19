# PyUncertainNumber

<!-- ```{image} _static/hex.png
:alt: The Py-Pkgs-Cookiecutter logo.
:width: 250px
:align: center
``` -->

<br>

`PyUncertainNumber` is a Python function library for generic computational tasks focussing on aleatory and epistemic uncertainty, which provides a research-grade computing environment for uncertainty characterisation, propagation, validation and uncertainty extrapolation.

# PyUncertainNumber

PyUncertainNumber is an generic computational tasks focusing on uncertainty quantification. 
It provides a research-grade computing environment for uncertainty characterisation, propagation, validation 
and uncertainty extrapolation. Four key modules comprised of the UQ ecosystem: 
Uncertainty characterisation module, Uncertainty propagation module, Uncertainty validation module and Uncertainty extrapolation module.


## Usage

`PyUncertainNumber` can be used to easily create a `PBox` or an `Interval` object:

```python
from PyUncertainNumber.UN import UncertainNumber

pbox_ex = UncertainNumber(
    name='elas_modulus', symbol='E', units='KPa', essence='distribution', 
    distribution_initialisation=['uniform', [(0,1),(1,2)]])
pbox_ex._math_object.show()
```


## Installation of the development version

```bash
$ pip install -e .
```

## Contributing

Interested in contributing? Check out the contributing guidelines. 
Please note that this project is released with a Code of Conduct. 
By contributing to this project, you agree to abide by its terms.

## License

`PyUncertainNumber` was created by Yu Chen (Leslie). It is licensed under the terms
of the MIT license.


## Features

```{include} stubs/features-stub.md
```

## Quick Start

```{include} stubs/quickstart-stub.md
```

<!-- ## Parameters

```{include} stubs/parameters-stub.md
``` -->

## Learn More

To learn more, checkout the sections below.

```{eval-rst}
.. container:: button

   :doc:`Quick Start <quickstart>` :doc:`User Guide <user-guide>`
   `Python Packages Book <https://py-pkgs.org/>`_ :doc:`Contributing Guidelines <contributing>`
```

## Acknowledgements

`PyUncertainNumber` was originally developed for use in the DAWS2 project. It has the capacity serve as a full-fleged UQ softwware to work beyond to fulfill general UQ challenges.

```{toctree}
:maxdepth: 2
:hidden:

Home <self>
Quick Start <quickstart>
User Guide <user-guide>
Contributing <contributing>
Code of Conduct <conduct>
<!-- Python Packages Book <http://py-pkgs.org> -->
```
