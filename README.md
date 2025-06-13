![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyuncertainnumber)
![version](https://img.shields.io/pypi/v/pyuncertainnumber)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15658422.svg)](https://doi.org/10.5281/zenodo.15658422)
![Documentation Status](https://readthedocs.org/projects/pyuncertainnumber/badge/?version=latest)
![license](https://img.shields.io/github/license/leslieDLcy/PyUncertainNumber)
![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

<p align="center">
  <img src="./assets/UNlogo3.png" alt="Logo" width="200"/>
</p>


# PyUncertainNumber

<!-- some banners -->

<!-- <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a> -->

**Uncertain Number** refers to a class of mathematical objects useful for risk analysis that generalize real numbers, intervals, probability distributions, interval bounds on probability distributions (i.e. [probability boxes](https://en.wikipedia.org/wiki/Probability_box)), and finite DempsterShafer structures. Refer to the [informative documentation](https://pyuncertainnumber.readthedocs.io/en/latest/index.html) for additional details.

## quick start

`PyUncertainNumber` can be used to easily create an `UncertainNumber` object, which may embody a mathematical construct such as `PBox`, `Interval`, `Distribution`, or `DempsterShafer` structure.

```python
from pyuncertainnumber import UncertainNumber as UN

e = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa', 
    essence='pbox', 
    distribution_parameters=['gaussian', ([0,12],[1,4])])
```

<!-- add some pbox plots herein -->
<img src="./assets/myAnimation.gif" alt="drapbox dynamic visualisationwing" width="500"/>

## installation

**Requirement:** It requires `Python >=3.11`

`PyUncertainNumber` can be installed from [PyPI](https://pypi.org/project/pyuncertainnumber/). Upon activation of your virtual environment, use the code below in your terminal. For additional instructions, refer to [installation guide](https://pyuncertainnumber.readthedocs.io/en/latest/guides/installation.html).

```shell
pip install pyuncertainnumber
```

## features

- `PyUncertainNumber` is a Python package for generic computational tasks focussing on rigourou uncertainty analysis, which provides a research-grade computing environment for uncertainty characterisation, propagation, validation and uncertainty extrapolation.
- `PyUncertainNumber` supports probability bounds analysis to rigorously bound the prediction for the quantity of interest with mixed uncertainty propagation.
- `PyUncertainNumber` also features great natural language support as such characterisatin of input uncertainty can be intuitively done by using natural language like `about 7` or simple expression like `[15 +- 10%]`, without worrying about the elicitation.
- features the save and loading of UN objects
- yields much informative results such as the combination that leads to the maximum in vertex method.

## UQ multiverse

UQ is a big world (like Marvel multiverse) consisting of abundant theories and software implementations on multiple platforms. We focus mainly on the imprecise probability frameworks. Some notable examples include [OpenCossan](https://github.com/cossan-working-group/OpenCossan) [UQlab](https://www.uqlab.com/) in Matlab and [ProbabilityBoundsAnalysis.jl](https://github.com/AnderGray/ProbabilityBoundsAnalysis.jl) in Julia, and many others of course. `PyUncertainNumber` builds upon on a few pioneering projects and will continue to be dedicated to support imprecise analysis in engineering using Python.

<!-- ## Contributing

Interested in contributing? Check out the contributing guidelines. 
Please note that this project is released with a Code of Conduct. 
By contributing to this project, you agree to abide by its terms. -->

<!-- ## License

`PyUncertainNumber` was created by Yu Chen (Leslie). It is licensed under the terms
of the MIT license. -->
