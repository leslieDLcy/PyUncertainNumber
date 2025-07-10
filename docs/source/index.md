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
```

```{toctree}
:caption: probability bounds analysis
:hidden:

pbox
cbox
interval_analysis
```

```{toctree}
:caption: API references
:hidden:
autoapi/index
```

# PyUncertainNumber

<br>

**Uncertain Number** refers to a class of mathematical objects useful for risk analysis that generalize real numbers, [intervals](https://en.wikipedia.org/wiki/Interval_arithmetic), probability distributions, interval bounds on probability distributions (i.e. [probability boxes](https://en.wikipedia.org/wiki/Probability_box)), and [finite DempsterShafer structures](https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory#:~:text=Often%20used%20as%20a%20method,on%20independent%20items%20of%20evidence.).

## features

- `PyUncertainNumber` is a Python package for generic computational tasks focussing on rigourou uncertainty analysis, which provides a research-grade computing environment for uncertainty characterisation, propagation, validation and uncertainty extrapolation.
- `PyUncertainNumber` supports probability bounds analysis to rigorously bound the prediction for the quantity of interest with mixed uncertainty propagation.
- `PyUncertainNumber` also features great natural language support as such characterisatin of input uncertainty can be intuitively done by using natural language like `about 7` or simple expression like `[15 +- 10%]`, without worrying about the elicitation.
- features the save and loading of UN objects
- yields much informative results such as the combination that leads to the maximum in vertex method.

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

Many propagation methodologies are provided to rigorously propgate the uncertainty through the computational pipeline, intrusively or non-intrusively.

```python

# shortcut to create uncertain numbers
a = pun.normal([2,3], [1])
b = pun.normal([10,14], [1])

# specify a response function
def foo(x): return x[0] ** 3 + x[1] + x[2]

# intrusive call signature which allows for drop-in replacements
response = foo([a, b])

# alternatively, one can use a more generic call signature
p = Propagation(vars=[a, b], func=foo, method='slicing', interval_strategy='direct')
response = p.run(n_slices=50)
```

```{attention}
The libary is under active develpment, so APIs will change across different versions.
```

## installation

```{tip}
- See [installation](./guides/installation.md) for additional details.
- **Requirement:** Python >=3.11
```

`PyUncertainNumber` can be installed from [PyPI](https://pypi.org/project/pyuncertainnumber/). Upon activation of your virtual environment, use your terminal. While we'd like to refer to the library as `PyUncertainNumber`in PascalCase, we use all lowercase (i.e. pyuncertainnumber) when installing from [PyPI](https://pypi.org/project/pyuncertainnumber/) following [PEP 8](https://peps.python.org/pep-0008/) convention.
For additional details, refer to [installation guide](https://pyuncertainnumber.readthedocs.io/en/latest/guides/installation.html).

```shell
pip install pyuncertainnumber
```

## UQ multiverse

UQ is a big world (like Marvel multiverse) consisting of abundant theories and software implementations on multiple platforms. Some notable examples include [OpenCossan](https://github.com/cossan-working-group/OpenCossan), [UQlab](https://www.uqlab.com/) in Matlab and [ProbabilityBoundsAnalysis.jl](https://github.com/AnderGray/ProbabilityBoundsAnalysis.jl) in Julia, and many others of course. `PyUncertainNumber` builds upon on a few pioneering projects and will continue to be dedicated to support imprecise analysis in engineering using Python.

## Acknowledgements

`PyUncertainNumber` was originally developed for use in the DAWS2 project. It has the capacity serve as a full-fleged UQ softwware to work beyond to fulfill general UQ challenges.
