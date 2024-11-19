# DAWS2

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

PyUncertainNumber
--------

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