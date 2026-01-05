<p align="center">
  <img src="./assets/UNlogo3.png" alt="Logo" width="200"/>
</p>

![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![version](https://img.shields.io/pypi/v/pyuncertainnumber)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17235456.svg)](https://doi.org/10.5281/zenodo.17235456)
![Documentation Status](https://readthedocs.org/projects/pyuncertainnumber/badge/?version=latest)
![license](https://img.shields.io/github/license/leslieDLcy/PyUncertainNumber)
![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

Scientific computations are surrounded by various forms of uncertainty, requiring appropriate treatment to maximise the credibility of computations. Empirical information is often scarce, vague, conflicting and imprecise, requiring expressive uncertainty structures for trustful representation, aggregation and propagation.

This package is underpinned by a framework of ***uncertain number*** which allows for a closed computation ecosystem whereby trustworthy computations can be conducted in a rigorous manner. <ins>It provides capabilities across the typical uncertainty analysis pipeline, encompassing characterisation, aggregation, propagation, model updating, and applications including reliability analysis and optimisation under uncertainty, especially with a focus on imprecise probabilities.</ins>

> ***Uncertain Number*** refers to a generalised representation that unifies several uncertainty constructs including real numbers, intervals, probability distributions, interval bounds on probability distributions (i.e. [probability boxes](https://en.wikipedia.org/wiki/Probability_box)), and [finite DempsterShafer structures](https://en.wikipedia.org/wiki/Dempster–Shafer_theory). It is mostly suitable for managing mixed types of uncertainties.



## Getting started

Explore the [documentation](https://pyuncertainnumber.readthedocs.io/en/latest/index.html) to get started, featuring hands-on [tutorials](https://pyuncertainnumber.readthedocs.io/en/latest/tutorials/index.html) and in-depth [examples](https://pyuncertainnumber.readthedocs.io/en/latest/examples/index.html) that showcase the power of the package.


>`pyuncertainnumber` [exposes APIs at different levels](file:///Users/lesliec/Documents/Github_repos/pyuncertainnumber/docs/_build/html/tutorials/getting_started.html). It features **high-level APIs** best suited for new users to quickly start with uncertainty computations with [*uncertain numbers*](https://pyuncertainnumber.readthedocs.io/en/latest/tutorials/what_is_un.html), and also **low-level APIs** allowing experts to have additional controls over mathematical constructs such as p-boxes, Dempster Shafer structures, probability distributions, etc.


## Installation

`PyUncertainNumber` can be installed from [PyPI](https://pypi.org/project/pyuncertainnumber/). Upon activation of your virtual environment, use the code below in your terminal. For additional instructions, refer to [installation guide](https://pyuncertainnumber.readthedocs.io/en/latest/guides/installation.html).

```shell
pip install pyuncertainnumber
```

## Capabilities

<p align="center">
  <img src="./assets/up_flowchart.png" alt="Logo" width="1000"/>
</p>

- `PyUncertainNumber` is a Python package for generic computational tasks focussing on **rigorous uncertainty analysis**, which provides a research-grade computing environment for uncertainty characterisation, propagation, validation and uncertainty extrapolation.
- `PyUncertainNumber` supports [probability bounds analysis](https://en.wikipedia.org/wiki/Probability_bounds_analysis) to rigorously bound the prediction for the quantity of interest with mixed uncertainty propagation.
- `PyUncertainNumber` also features great **natural language support** as such characterisatin of input uncertainty can be intuitively done by using natural language like `about 7` or simple expression like `[15 +- 10%]`, without worrying about the elicitation.
- Interoperability via serialization: features the save and loading of Uncertain Number objects to work with downstream applications.
- Yields informative results during the computation process such as the combination that leads to the maximum in vertex method.

## UQ multiverse

UQ is a big world (like Marvel multiverse) consisting of abundant theories and software implementations on multiple platforms. Some notable examples include [OpenCossan](https://github.com/cossan-working-group/OpenCossan) [UQlab](https://www.uqlab.com/) in Matlab and [ProbabilityBoundsAnalysis.jl](https://github.com/AnderGray/ProbabilityBoundsAnalysis.jl) in Julia, and many others of course. We focus mainly on the imprecise probability frameworks. `PyUncertainNumber` is rooted in Python and has close ties with the Python scientific computing ecosystem, it builds upon and greatly extends a few pioneering projects, such as [intervals](https://github.com/marcodeangelis/intervals), [scipy-stats](https://docs.scipy.org/doc/scipy/tutorial/stats.html) and [pba-for-python](https://github.com/Institute-for-Risk-and-Uncertainty/pba-for-python) to generalise probability and interval arithmetic. Beyond arithmetic calculations, `PyUncertainNumber` has offered a wide spectrum of algorithms and methods for uncertainty characterisation, propagation, surrogate modelling, and optimisation under uncertainty, allowing imprecise uncertainty analysis in both intrusive and non-intrusive manner. `PyUncertainNumber` is under active development and will continue to be dedicated to support imprecise analysis in engineering using Python.


## Citation

> [Yu Chen, Scott Ferson (2025). Imprecise uncertainty management with uncertain numbers to facilitate trustworthy computations.](https://proceedings.scipy.org/articles/ahrt5264), SciPy proceedings 2025.

A downloadable version can be accessed [here](https://www.researchgate.net/publication/396633010_Imprecise_uncertainty_management_with_uncertain_numbers_to_facilitate_trustworthy_computations).

``` bibtex
@inproceedings{chen2025scipyproceed,
  title = {Imprecise uncertainty management with uncertain numbers to facilitate trustworthy computations},
  booktitle = {SciPy Proceedings},
  year = {2025},
  author = {Chen, Yu and Ferson, Scott},
  doi = {10.25080/ahrt5264}
}

@software{chen_2025_17235456,
  author       = {Chen, (Leslie) Yu},
  title        = {PyUncertainNumber},
  publisher    = {Zenodo},
  version      = {0.1.1},
  doi          = {10.5281/zenodo.17235456},
  url          = {https://doi.org/10.5281/zenodo.17235456},
}
```

<!-- ## Contributing

Interested in contributing? Check out the contributing guidelines. 
Please note that this project is released with a Code of Conduct. 
By contributing to this project, you agree to abide by its terms. -->

<!-- ## License

`PyUncertainNumber` was created by Yu Chen (Leslie). It is licensed under the terms
of the MIT license. -->
