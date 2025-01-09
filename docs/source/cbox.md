# Confidence box

Trationally confidence intervals are widely used in engineering to making inference with respect to parameters of interest. It is appealing as it provides a gurantee of statistical performance through repeated use. Studies in frequetist inference continue to wonder the possibility of a distributional estimator just like the Bayesian posterior, which leads to a lot of developments towards the **confidence distribution** though it is not technically the same as a distribution in the canonical sense [^1]. It conveiently provides confidence intervals of all levels for a parameter of interest. Confidence boxes, or confidence structures, (abbreviated as c-box) are imprecise generalisations of confidence distributions and they can be applied to problems with discrete observatons, interval-censored data, and even inference problems in which no assumption about the distribution shape can be made.

The figure below provides an illustration of the confidence box which yields several confidence intrevals for the parameter $\theta$.

<!-- <p float="left">
  <img src="../../assets/left.png" width="32% />
  <img src="./../assets/middle.png" width="32% />
  <img src="./../assets/right.png" width="32% />
</p> -->

```{image} ../../assets/confidence_distribution_illustration.png
:alt: confidence distribution illustration
:class: bg-primary
:width: 1000px
:align: center

Illustration of confidence distribution
```

<!-- exmaple normal confidence distribution -->

Generallly one cannot directly compute with confidence intervals, but you can compute with confidence boxes from which you can get arbitrary confidence intervals for the results.

---
[^1]: A confidence distribution is merely a ciphering device that encodes confidence intervals for each possible confidence level.
