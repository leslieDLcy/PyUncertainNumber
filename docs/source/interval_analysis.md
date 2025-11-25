# interval analysis

Intervals play a central role in the [probability bounds analysis](https://en.wikipedia.org/wiki/Probability_bounds_analysis) and has been discussed in the context of computational functional analysis. It is known for making rigourous computations in terms of three kinds of errors in numerical analysis: rounding errors, truncation errors, and input errors.

### Interval arithmetic

```{hint}
the key point of the definitions of basic arithmetic operations between intervals is **computing with intervals is computing with sets**.
```

Interval arithmetic operations can be defined as :

$$
X \odot Y = \{ x \odot  y : x \in X, y \in Y \}
$$
where $\odot$ stands for the elementary binary operations such as addition or product etc.

A key consideration of the propagation of interval objects is the *dependency* issue, which hinders the naive uses of interval arithmetic in many problems as it often yields inflated interval outputs. The *image set* under a real-valued function mapping $f$ as $x$ varies through a given interval $[X]$ (or simply $X$) can be defined as:

```{note}
Square backets are used to visually hint the nature of an Interval typed variable. In Python, square brackets suggest a list datatype which is ubiquitous, as such, in `PyUncertainNumber` we provide a parser for easy creation of interval objects with lists.
```

$$
f(X) = \{ f(x): x \in X \}
$$

It should be noted that when interval arithmetic is naively used in computations, it may not necessary yield the best-possible (or sharpest) range. A useful interval extension is the **mean value form**

$$
f(X) \subseteq F_{MV}(X) = f(m) + \sum_{i=1}^{n} D_{i}F(X) (X_{i} - m_{i})
$$

where $X$ denotes an interval vector and $m$ is the midpoint vector while $D_{i}F$ be an interval extension of the first derivatives $\partial f / \partial x_{i}$.

Refer to {ref}`propagation_guide` page for additional methods (e.g. vertex method, interval substitution) for interval propagation.


### handling with measurement uncertainty

Naturally, intervals serve as an intuitive representations for measurement error. Engineers often report measurement incertitude in the form of $[m \pm w]$. Many statistical models arise based on the interval statistics. As an example, similar to using a probability distribution to characterise precise data, we can employ a p-box to characterise interval-valued data. Generalisation of empirical cumulative distribution function can also be intuitively made. In addition, Kolmogorov–Smirnov bounds can also be generalised to derive confidence limits for imprecise data.

$$
[\overline{F}(x), \underline{F}(x)] = \big[ \min(1, \hat{F}_{L}(x) + D_{N}^{\alpha}), \ \max(0, \hat{F}_{R}(x) - D_{N}^{\alpha}) \big]
$$

`PyUncertainNumber` provides a straghtforward syntax in charactersing interval-valued data.

```{see also}
characterisation
```
