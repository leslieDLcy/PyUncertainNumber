# Uncertainty characterisation
<!-- write something about uncertainty characterisation -->

### variability and incertitude

Partial specification given information, bla bla ..

````{tip}
It is suggested to use interval analysis for propagating ignorance and the methods of probability theory for propagating variability.

```{seealso}
propagation
```
````

### bounding distributional parameters

The mean of a normal distribution may be elicited from an expert but this expert cannot be precise to a certain value but rather give a range based on past experience.

````{tab} verbose
To comprehensively characterise a pbox, specify the bounds for the parameters along with many other ancillary fields.

```python
from pyuncertainnumber import UncertainNumber as UN

e = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa', 
    essence='pbox', 
    distribution_parameters=['gaussian', ([0,12],[1,4])])
```
````

````{tab} shortcut
In cases where one wants to do computations quickly.

```python
import pyuncertainnumber as pun
un = pun.norm([0,12],[1,4])
```
````

### aggregation of multiple sources of information

Expert elicitation has been a challenging topic, especially when knowledge is limited and measurements are sparse. Multiple experts may not necessarily agree on the choice of elicited prbability distributions, which leads to the need for aggregation. Below shows two situations for illustration.

Assume the expert opinions are expressed in closed intervals. There may well be multiple such intervals from different experts and these collections of interval can be overlapping, partially contradictory or even completely contradictory. Their relative credibility may be expressed in probabilities. Essentially such information creates a **Dempster-Shafer structure**. On the basis of a mixture operation, such information can be aggregated into a **pbox**.

```{tip}
The different sub-types of uncertain number can normally convert to one another (though may not be one by one), ergo the uncertain number been said to be a unified representation.
```

Pbox arithmetic also extends the convolution of probability distributions which has typically been done with the independence assumption. However, often in engineering modelling practices independence is assumed for mathematical easiness rather than warranted. Fortunately, the uncertainty about the dependency between random variables can be characterised by the probability bounds, as seen below. It should be noted that such dependency bound does not imply independence.

```{image} ../../../assets/addition_bound.png
:alt: sum of two random variables without dependency specification
:class: bg-primary
:width: 600px
:align: center

The sum of two random variables of lognormal distribution without dependency specification
```

### known statistical properties

When the knowledge of a quantity is limited to the point where only some statistical information is available, such as the *min*, *max*, *median* etc. but not about the distribution and parameters, such partial information can serve as **constraints** to bound the underlying distribution:

```{image} ../../../assets/known_constraints.png
:alt: known constraints
:class: bg-primary
:width: 1000px
:align: center
```

### hedged numerical expression

Sometimes only purely qualitive information is available. An important part of processing elicited numerical inputs is an ability to quantitatively decode natural-language words, the linguistic information, that are commonly used to express or modify numerical values. Some example include ‘about’, ‘around’, ‘almost’, ‘exactly’, ‘nearly’, ‘below’, ‘at least’, ‘order of’, etc. A numerical expression with these approximators are called *hedges*. Extending upon the significant-digit convention, a series of interval interpretations of common hedged numerical expressions are proposed.

```{image} ../../../assets/interval_hedge.png
:alt: interval hedges
:class: bg-primary
:width: 1000px
:align: center

Symmetric (left) and asymmetric (right) approximators of the number 7
```

Besides intervals, `PyUncertainNumber` also supports interpreting hedged expressions into p-boxes. As an example, assume one wants to find out what "about" is about in terms of the uncertainty. The syntax and result is shown below:

```python
import pyuncertainnumber as pun
pun.hedge_interpret('about 200', return_type='pbox').display()
```

```{image} ../../../assets/about_200.png
:alt: about 200
:class: bg-primary
:width: 600px
:align: center

hedged numerical expression "about 200"
```

### interval measurements
