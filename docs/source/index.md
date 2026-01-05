```{toctree}
:maxdepth: 2
:hidden:
Home <self>
```

```{toctree}
:caption: User Guides
:hidden:

guides/installation
guides/uc
guides/up
pbox
cbox
interval_analysis
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
:titlesonly:
:hidden:

tutorials/index
```

```{toctree}
:caption: Examples
:maxdepth: 1
:titlesonly:
:hidden:

examples/index
```

```{toctree}
:caption: API references
:hidden:

autoapi/index
```

# PyUncertainNumber

<br>

This package is underpinned by a framework of **uncertain numbers** which allows for a closed computation ecosystem whereby trustworthy computations can be conducted in a rigorous manner. <ins>It provides capabilities across the typical uncertainty analysis pipeline, encompassing uncertainty characterisation, aggregation, propagation, model updating, and applications including reliability analysis and optimisation under uncertainty, especially with a focus on imprecise probabilities</ins>.

```{note}
**Uncertain Number** refers to a class of mathematical objects useful for risk analysis that generalize real numbers, intervals, probability distributions, interval bounds on probability distributions (i.e. [probability boxes](https://en.wikipedia.org/wiki/Probability_box)), and [finite DempsterShafer structures](https://en.wikipedia.org/wiki/Dempster–Shafer_theory). Refer to the [source code repository](https://github.com/leslieDLcy/PyUncertainNumber) of this package for additional introduction.
```

## Capabilities

<p align="center">
  <img src="./_static/up_flowchart.png" alt="Logo" width="1000"/>
</p>

- `PyUncertainNumber` is a Python package for generic computational tasks focussing on **rigorous uncertainty analysis**, which provides a research-grade computing environment for uncertainty characterisation, propagation, validation and uncertainty extrapolation.
- `PyUncertainNumber` supports [probability bounds analysis](https://en.wikipedia.org/wiki/Probability_bounds_analysis) to rigorously bound the prediction for the quantity of interest with mixed uncertainty propagation.
- `PyUncertainNumber` also features great **natural language support** as such characterisation of input uncertainty can be intuitively done by using natural language like `about 7` or simple expression like `[15 +- 10%]`, without worrying about the elicitation.
- Interoperability via serialization: features the save and loading of Uncertain Number objects to work with downstream applications.
- Yields informative results during the computation process such as the combination that leads to the maximum in vertex method.


```{tip}
`pyuncertainnumber` exposes APIs at different levels. It features high-level APIs best suited for new users to quickly start with uncertainty computations with [*uncertain numbers*], and also low-level APIs allowing experts to have additional controls over mathematical constructs such as p-boxes, Dempster Shafer structures, probability distibutions, etc.
```

<!-- ## latest features -->

## Quick start: uncertainty characterisation and propagation

`PyUncertainNumber` can be used to easily create an `UncertainNumber` object, which may embody a mathematical construct such as `PBox`, `Interval`, `Distribution`, or `DempsterShafer` structure.

```python
from pyuncertainnumber import UncertainNumber as UN
import pyuncertainnumber as pun

# shortcut to create uncertain numbers
a = pun.normal([2,3], [1])
b = pun.normal([10,14], [1])
```

<!-- add some pbox plots herein -->
<img src="./_static/myAnimation.gif" alt="drapbox dynamic visualisationwing" width="500"/>


`PyUncertainNumber` supports a duck-typing way of doing probability arithmetic, such that users can directly compute with uncertain numbers through a Python function by drop-in replacements as if they were real numbers.

```python
a - b * a + b**2
```

Besides, [many propagation methodologies](https://pyuncertainnumber.readthedocs.io/en/latest/guides/up.html) are provided to rigorously propagate the uncertainty through the computational pipeline, intrusively or non-intrusively.

```python
# specify a response function
def foo(x): return x[0] ** 3 + x[1] + 2

# duck-typing signature which allows for drop-in replacements
response = foo([a, b])

# alternatively, one can use a more generic call signature suitable for black-box models
p = pun.Propagation(vars=[a, b], func=foo, method='slicing', interval_strategy='direct')
response = p.run(n_slices=50)
```

```{attention}
The libary is under active develpment, so APIs will change across different versions.
```

```{tip}
If looking for deeper controls and customisation, refer to the [Low-level `pba` APIs](https://pyuncertainnumber.readthedocs.io/en/latest/tutorials/getting_started.html#low-level-pba-apis)
 for advanced usage.
```


## Installation

```{note}
- See [installation](./guides/installation.md) for additional details.
- **Requirement:** Python >=3.11
```

`PyUncertainNumber` can be installed from [PyPI](https://pypi.org/project/pyuncertainnumber/). Upon activation of your virtual environment, use your terminal. While we'd like to refer to the library as `PyUncertainNumber`in PascalCase, we use all lowercase (i.e. pyuncertainnumber) when installing from [PyPI](https://pypi.org/project/pyuncertainnumber/) following [PEP 8](https://peps.python.org/pep-0008/) convention.
For additional details, refer to [installation guide](https://pyuncertainnumber.readthedocs.io/en/latest/guides/installation.html).

```shell
pip install pyuncertainnumber
```

## UQ multiverse

UQ is a big world (like Marvel multiverse) consisting of abundant theories and software implementations on multiple platforms. Some notable examples include [OpenCossan](https://github.com/cossan-working-group/OpenCossan) [UQlab](https://www.uqlab.com/) in Matlab and [ProbabilityBoundsAnalysis.jl](https://github.com/AnderGray/ProbabilityBoundsAnalysis.jl) in Julia, and many others of course. We focus mainly on the imprecise probability frameworks. `PyUncertainNumber` is rooted in Python and has close ties with the Python scientific computing ecosystem, it builds upon and greatly extends a few pioneering projects, such as [intervals](https://github.com/marcodeangelis/intervals), [scipy-stats](https://docs.scipy.org/doc/scipy/tutorial/stats.html) and [pba-for-python](https://github.com/Institute-for-Risk-and-Uncertainty/pba-for-python) to generalise probability and interval arithmetic. Beyond arithmetic calculations, `PyUncertainNumber` has offered a wide spectrum of algorithms and methods for uncertainty characterisation, propagation, surrogate modelling, and optimisation under uncertainty, allowing imprecise uncertainty analysis in both intrusive and non-intrusive manner. `PyUncertainNumber` is under active development and will continue to be dedicated to support imprecise analysis in engineering using Python.

## Acknowledgements

`PyUncertainNumber` was originally developed for use in the DAWS2 project. It has the capacity to serve as a full-fleged UQ softwware to work beyond to fulfill general UQ challenges.


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