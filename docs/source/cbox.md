# Confidence box

Traditionally [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval) is widely used in engineering to make inference with respect to parameters of interest. It is appealing as it provides a gurantee of statistical performance through repeated use. Studies in frequetist inference continue to wonder the possibility of a *distributional estimator* just like the Bayesian posterior, which leads to significant developments towards the **confidence distribution**, though it is not technically the same as a distribution in the canonical sense.

```{tip}
A confidence distribution is essentially a ciphering device that encodes confidence intervals for each possible confidence level.
```

The confidence distribution conveniently provides confidence intervals of all levels for a parameter of interest. **Confidence boxes**, or confidence structures[^1], (abbreviated as c-box) are imprecise generalisations of confidence distributions and they can be applied to problems with discrete observatons, interval-censored data, and even inference problems in which no assumption about the distribution shape can be made.

The figure below provides an illustration of the confidence box which yields several confidence intrevals for the parameter $\theta$.

```{image} ../../assets/confidence_distribution_illustration.png
:alt: confidence distribution illustration
:class: bg-primary
:width: 1000px
:align: center

Illustration of confidence distribution
```

```{admonition} An example of Gaussian distributed parameters
The cumulative confidence distribution for $\mu$ can be defined as:

$$
H(\mu, \mathbf{x}) = F_{t_{n-1}} \Big( \frac{\mu - \bar{x}}{s_{x}/\sqrt{n}} \Big)
$$

and also the distributional estimator for the parameter $\sigma^2$ is:

$$
H_{\chi^{2}}(\sigma^2, \mathbf{x}) = 1 - F_{\chi_{n-1}^2} \Big( \frac{(n-1) s_{x}^2}{\sigma^2} \Big)
$$

where $F_{t_{n-1}}$ is the c.d.f for a $t$ distribution with $n-1$ degrees of freedom while $F_{\chi_{n-1}^2}$ is the c.d.f of the $\chi_{n-1}^2$ distribution
```

<!-- exmaple normal confidence distribution -->

Generallly one cannot directly compute with confidence intervals, but you can compute with confidence boxes from which you can get arbitrary confidence intervals for the results. To facilitate the analysis, `PyUncertainNumber` provides straightforward syntax to derive the sample-dependent confidence box, as shown below:

```python
# x[i] ~ binomial(N, p)
# k=2, n=10
c = infer_cbox('binomial', data=[2], N=10)
```

```{image} ../../assets/cbox_illustration.png
:alt: confidence box example with Binomial distribution
:class: bg-primary
:width: 600px
:align: center
```

[^1]: Balch, Michael Scott. "Mathematical foundations for a theory of confidence structures." International Journal of Approximate Reasoning 53.7 (2012): 1003-1019.
