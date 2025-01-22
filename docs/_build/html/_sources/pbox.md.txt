# Probability box

A conspicuous problem in probability distribution elicition, for example in probabilistic modelling analysis, is that the specification is typically precise, despite hardly justified by empirical informtion in many cases.

```{attention}
epistemic uncertainy remains on the shape, parameters, and dependencies of the distributions.
```

[Probability box](https://en.wikipedia.org/wiki/Probability_box) (abbreviated as `p-box`) essentially represents bounds on the cumulative distribution function (c.d.f) of the underlying random variable. Let $[\overline{F}, \underline{F}]$ denotes the **set** of all nondecreasing functions from the reals into $[0,1]$ such that $\underline{F} \le F \le \overline{F}$. This means that, $[\overline{F}, \underline{F}]$ denotes a p-box for a random varaible $X$ whose c.d.f $F$ is unknown except that it is within the "box" circumscribed by the lower ($\underline{F}$) and upper bound ($\overline{F}$).

$$
\underline{F} \le F(x) \le \overline{F}
$$

p-box collectively reflects the variability (aleatoric uncertainty) and incertitude (epistemic uncertainty) in one structure for the uncertain quantity of interest. The horizontal span of the probabilty bounds are a function of the variability and the vertical breadth of the bounds is a function of ignorance.

<!-- a plot -->
```{image} ../../assets/pbox_illustration.png
:align: center
```

```{hint}
There is a storng link between p-box and Dempster-Shafer structures (which `PyUncertainNumber` also explicitly provides support :boom:). Each can be converted to the other. However, it should be noted such translation is not one-to-one.
```

`PyUncertainNumber` provides support for operations with p-boxes ranging from [characterisation](./guides/uc.md), aggregation, propagation. Go check out these links for details as to the computation with p-boxes. Meanwhile, quick examples show below:

```python
from PyUncertainNumber import UncertainNumber as UN

un = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa', 
    essence='pbox', 
    distribution_parameters=['gaussian', [(0,12),(1,4)]])
_ = un.display(style='band')
```

<!-- a plot -->
```{image} ../../assets/myAnimation.gif
:align: center
:width: 600px
```

<!-- the idea is for motivating purposes not to educate them. -->
<!-- Content can have inline markup like *emphasis*, **strong emphasis**,
`inline literals`, {sub}`subscript`, {sup}`superscript` and so much more.
Providing a reference to {pep}`8` is straightforward. You can also include
abbreviations like {abbr}`HTML (Hyper Text Markup Language)`.

> This is blockquoted text.

It is possible to have multiple paragraphs of text, which get separated
from each other visually. When stronger visual separation is desired, a
horizontal separator can be used (3 or more punctuation characters on a line).

---

This is written in Markdown. -->

<!--1.  when there is limited knowledge in terms of characterisation of input variables -->

<!-- 2. aggregating expert judgement -->