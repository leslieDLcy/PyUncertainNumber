<p align="center">
  <img src="./assets/UNlogo3.png" alt="Logo" width="200"/>
</p>

![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![version](https://img.shields.io/pypi/v/pyuncertainnumber)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15658422.svg)](https://doi.org/10.5281/zenodo.15658422)
![Documentation Status](https://readthedocs.org/projects/pyuncertainnumber/badge/?version=latest)
![license](https://img.shields.io/github/license/leslieDLcy/PyUncertainNumber)
![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)


# PyUncertainNumber

<!-- some banners -->

<!-- <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a> -->
Scientific computations of complex systems are surrounded by various forms of uncertainty,  requiring appropriate treatment to maximise the credibility of computations. Empirical information for characterisation is often scarce, vague, conflicting and imprecise, requiring expressive uncertainty structures for trustful representation, aggregation and propagation.

This package is underpined by a framework of **uncertain number** which allows for a closed computation ecosystem whereby trustworthy computations can be conducted in a rigorous manner. It provides capabilities across the typical uncertainty analysis pipeline, encompassing characterisation, aggregation, propagation, and applications including reliability analysis and optimisation under uncertainty, especailly with a focus on imprecise probabilities.

**Uncertain Number** refers to a class of mathematical objects useful for risk analysis that generalize real numbers, intervals, probability distributions, interval bounds on probability distributions (i.e. [probability boxes](https://en.wikipedia.org/wiki/Probability_box)), and [finite DempsterShafer structures](https://en.wikipedia.org/wiki/Dempster–Shafer_theory). Refer to the [documentation](https://pyuncertainnumber.readthedocs.io/en/latest/index.html) of this package for additional introduction.

## Quick start

`PyUncertainNumber` can be used to easily create an `UncertainNumber` object, which may embody a mathematical construct such as `PBox`, `Interval`, `Distribution`, or `DempsterShafer` structure.

<!-- add some pbox plots herein -->
<img src="./assets/myAnimation.gif" alt="drapbox dynamic visualisationwing" width="500"/>

```python
from pyuncertainnumber import UncertainNumber as UN
import pyuncertainnumber as pun

# a verbose to instantiate uncertain numbers with ancillary fields
e = UN(
    name='elas_modulus', 
    symbol='E', 
    unit='Pa', 
    essence='pbox', 
    distribution_parameters=['gaussian', ([0,12],[1,4])])

# or simply use shortcut to create uncertain numbers
a = pun.normal([2,3], [1])
b = pun.normal([10,14], [1])

# specify a response function
def foo(x): return x[0] ** 3 + x[1] + 2

# intrusive call signature which allows for drop-in replacements
response = foo([a, b])

# alternatively, one can use a more generic call signature for propagation
p = pun.Propagation(vars=[a, b], func=foo, method='slicing', interval_strategy='direct')
response = p.run(n_slices=50)
```



## Installation

**Requirement:** `Python >=3.11`

`PyUncertainNumber` can be installed from [PyPI](https://pypi.org/project/pyuncertainnumber/). Upon activation of your virtual environment, use the code below in your terminal. For additional instructions, refer to [installation guide](https://pyuncertainnumber.readthedocs.io/en/latest/guides/installation.html).

```shell
pip install pyuncertainnumber
```

## Capabilities

- `PyUncertainNumber` is a Python package for generic computational tasks focussing on **rigourou uncertainty analysis**, which provides a research-grade computing environment for uncertainty characterisation, propagation, validation and uncertainty extrapolation.
- `PyUncertainNumber` supports [probability bounds analysis](https://en.wikipedia.org/wiki/Probability_bounds_analysis) to rigorously bound the prediction for the quantity of interest with mixed uncertainty propagation.
- `PyUncertainNumber` also features great **natural language support** as such characterisatin of input uncertainty can be intuitively done by using natural language like `about 7` or simple expression like `[15 +- 10%]`, without worrying about the elicitation.
- Interoperability via serialization: features the save and loading of Uncertain Number objects to work with downstream applications.
- Yields informative results during the computation process such as the combination that leads to the maximum in vertex method.

## UQ multiverse

UQ is a big world (like Marvel multiverse) consisting of abundant theories and software implementations on multiple platforms. We focus mainly on the imprecise probability frameworks. Some notable examples include [OpenCossan](https://github.com/cossan-working-group/OpenCossan), [UQlab](https://www.uqlab.com/) in Matlab and [UncertaintyQuantification.jl](https://github.com/FriesischScott/UncertaintyQuantification.jl), [ProbabilityBoundsAnalysis.jl](https://github.com/AnderGray/ProbabilityBoundsAnalysis.jl) in Julia, and many others of course. 
`PyUncertainNumber` is rooted in Python and has close ties with the Python scientific computing ecosystem, it builds upon and greatly extends a few pioneering projects, such as [intervals](https://github.com/marcodeangelis/intervals), [scipy-stats](https://docs.scipy.org/doc/scipy/tutorial/stats.html) and [pba-for-python](https://github.com/Institute-for-Risk-and-Uncertainty/pba-for-python) to generalise probability and interval arithmetic. Beyond arithmetics, `PyUncertainNumber` has offered a wide spectrum of algorithms and methods for uncertainty characterisation, propagation, surrogate modelling, and optimisation under uncertainty, allowing imprecise uncertainty analysis in both intrusive and non-intrusive manner. `PyUncertainNumber` is under active development and will continue to be dedicated to support imprecise analysis in engineering using Python.



<!-- ## Contributing

Interested in contributing? Check out the contributing guidelines. 
Please note that this project is released with a Code of Conduct. 
By contributing to this project, you agree to abide by its terms. -->

<!-- ## License

`PyUncertainNumber` was created by Yu Chen (Leslie). It is licensed under the terms
of the MIT license. -->
