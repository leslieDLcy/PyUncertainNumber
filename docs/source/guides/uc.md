# Uncertainty characterisation
<!-- write something about uncertainty characterisation -->

### variability and incertitude

Partial specification given information, bla bla ..

### bounding distributional parameters

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

Assume the expert opinions are expressed in closed intervals and their relative credibility are expresses in probabilities. Essentially such information creates a **Dempster-Shafer structure**. On the basis of a mixture operation, such information can be aggregated into a **pbox**.

```{tip}
The different sub-types of uncertain number can normally convert to one another (though may not be one by one), ergo the uncertain number been said to be a unified representation.
```

### hedged numerical expression

An important part of processing elicited numerical inputs is an ability to quantitatively decode natural-language words that are commonly used to express or modify numerical values. Some example include ‘about’, ‘around’, ‘almost’, ‘exactly’, ‘nearly’, ‘below’, ‘at least’, ‘order of’, etc. These are called *hedges*

### interval measurements
