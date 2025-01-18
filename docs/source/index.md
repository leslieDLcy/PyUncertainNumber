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
```

# PyUncertainNumber

<br>

**Uncertain Number** refers to a class of mathematical objects useful for risk analysis that generalize real numbers, [intervals](https://en.wikipedia.org/wiki/Interval_arithmetic), probability distributions, interval bounds on probability distributions (i.e. [probability boxes](https://en.wikipedia.org/wiki/Probability_box)), and [finite DempsterShafer structures](https://en.wikipedia.org/wiki/Dempster%E2%80%93Shafer_theory#:~:text=Often%20used%20as%20a%20method,on%20independent%20items%20of%20evidence.).

## features

- `PyUncertainNumber` is a Python package for generic computational tasks focussing on rigourou uncertainty analysis, which provides a research-grade computing environment for uncertainty characterisation, propagation, validation and uncertainty extrapolation.
- `PyUncertainNumber` also features great natural language support as such characterisatin of input uncertainty can be intuitively done by using natural language like `about 7` or simple expression like `[15 +- 10%]`, without worrying about the elicitation.
- features the save and loading of UN objects
- yields much informative results such as the combination that leads to the maximum in vertex method.

## quick start

`PyUncertainNumber` can be used to easily create a `PBox` or an `Interval` object:

```python
from pyuncertainnumber import UncertainNumber as UN

e = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa', 
    essence='pbox', 
    distribution_parameters=['gaussian', ([0,12],[1,4])])
```

<!-- add some pbox plots herein
```{image} ../../assets/myAnimation.gif
:alt: pbox dynamic visualisation
:class: bg-primary
:width: 600px
:align: center
``` -->

## installation

bla bla simple bla bla

```shell
pip install pyuncertainnumber
```

## UQ multiverse

UQ is a big world (like Marvel multiverse) consisting of abundant theories and software implementations on multiple platforms. Some notable examples include [OpenCossan](https://github.com/cossan-working-group/OpenCossan), [UQlab](https://www.uqlab.com/) in Matlab and [ProbabilityBoundsAnalysis.jl](https://github.com/AnderGray/ProbabilityBoundsAnalysis.jl) in Julia, and many others of course. `PyUncertainNumber` builds upon on a few pioneering projects and will continue to be dedicated to support imprecise analysis in engineering using Python.

## Acknowledgements

`PyUncertainNumber` was originally developed for use in the DAWS2 project. It has the capacity serve as a full-fleged UQ softwware to work beyond to fulfill general UQ challenges.

## API references
