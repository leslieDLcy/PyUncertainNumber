(propagation_guide)=
# Propagation

Methods to efficiently propagate different types of uncertainty through computational models are of vital interests.
`PyUncertainNumber` includes strategies for black box uncertainty propagation (i.e.non-intrusive) as well as a library of functions for PBA ([Probability Bounds Analysis](https://en.wikipedia.org/wiki/Probability_bounds_analysis)) if the code can be accessed (i.e. intrusive). `PyUncertainNumber` provides a series of uncertainty propagation methods.

It is suggested to use [interval analysis](../interval_analysis.md) for propagating ignorance and the methods of probability theory for propagating variability. But realistic engineering problems or risk analyses will most likely involve a mixture of both types and as such probability bounds analysis provides means to rigourously propagate the uncertainty.

For aleatory uncertainty, probability theory already provides some established approaches, such as Taylor expansion or sampling methods, etc. This guide will mostly focuses on the propagation of intervals due to the close relations with propagation of p-boxes. A detailed review can be found in this [report](https://sites.google.com/view/dawsreports/up/report). Importantly, see {ref}`up` for a hands-on tutorial.

## Vertex method

The vertex propagation method {cite:p}`DONG198765` is a straightforward way to project intervals through the code, by projecting a number of input combinations given by the Cartesian product of the interval bounds. This results in a total of $n=2d$ evaluations, where $d$ is the number of interval-valued parameters. In the case of two intervals, $[x]$ and $[y]$, the code, $f(·)$ must be evaluated four times at all endpoints. The main advantage of the method is its simplicity. But it should be used under the assumption of monotonicity.

## Subinterval reconstitution

To accommodate the presence of non-monotonic trends, the input intervals can be partitioned into smaller intervals, which can then be propagated through the model using vertex propagation and an output interval can be reassembled. The logic behind this method is that even though the code may not be monotonic over the full width of the interval, it will likely behave monotonically on a smaller interval, provided the underlying function is pathologically rough. Thus, output intervals for even highly non-linear functions can be computed to arbitrary reliability.

$$
f(X) = \cup_{j=1}^{n} f(X_{j}) \subseteq \cup_{j=1}^{n} F(X_{j})
$$

where $\cup_{j=1}^{n} X_{j} = X$ and $F$ denotes an interval extension of $f$.


## Cauchy-deviate method

Similar to the Monte Carlo methods, the Cauchy-deviate method {cite:p}`KREINOVICH2004267` is also built upon the direct sampling of uncertain input numbers expressed as intervals. This makes it suitable for propagating epistemic uncertainty through black-box models. For each uncertain number, the method generates random samples from a Cauchy distribution and computes the width of the output interval through appropriate scaling of these samples. If the model has more than one input, the errors in each are considered independent at the sampling step. By generating samples outside of the input interval bounds, the Cauchy-deviate method explicitly includes those bounds via normalisation, thus solving one of the problems classical sampling methods face when used to propagate epistemic uncertainty.

## Gradient-based optimisation methods

In general cases, interval propagation problem can be casted as an optimisation problem. These methods, gradients-based or gradients-free, would often start with a set of initial input values and by searching for new candidates which yield better solutions than the previous iterations find an optimum solution. Key characteristic of these methods is that, for each iteration, the search for better solution is local, in other words it takes place in the immediate neighbourhood of the input values used in the previous iteration, leading to a local optimum solution. 

```{image} ../../../assets/up_comparison.png
:alt: comparison of several propagation methods
:class: bg-primary
:width: 1000px
:align: center

Comparison of several propagation methods
```

## References

```{bibliography}
:filter: docname in docnames
```