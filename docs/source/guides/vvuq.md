(vvuq_guide)=
# VV&UQ framework

Numerical simulations plan an ever-growing role in simulating the behaviour of natural and engineered systems. Various sources uncertainties existing in the computational pipeline shadow the credibility the mathematical model or the numerical implementation {cite:p}`roy2011comprehensive`. 

```{note}
The VV&UQ framework is one comprehensive framework, composed by the elements of *validation*, *verification* and *uncertainty quantification*, that estimates the predictive uncertainty of computing applications.
```

## Validation, verification, and uncertainty quantification 

The uncertainty about the discrepancy between the model and reality is called *model form* uncertainty {cite:p}`cary2026summary`. This type of uncertainty is most commonly assessed during validation campaigns, which collect experimental data about the phenomena of interest and compare this data to model predictions of conditions nominally identical to those in the tests. Although reflecting the best state of knowledge of the engineer, experimental data is affected, among other things, by *measurement uncertainty*, which must be accounted for during validation.

Moreover, models are generally built to reflect a family of physical processes and are given tunable parameters which allow the engineer to set the model for use in specific scenarios. The values of these parameters which represent the true physical process, to be compared against in validation, are not precisely known, because of the presence of *parametric input uncertainty*. Models that compute numerical solutions to partial differential equations are also subject to *numerical uncertainty* due to finite spatial and/or temporal resolution, iterative convergence error, and the use of finite precision arithmetic in computer codes. 

Furthermore, the main purpose of any model is to be used where experiments are impractical or impossible to be conducted and the engineer must reason about the performance of the model from the evidence obtained during validation. This results in *extrapolation uncertainty*. The individual contributions of each uncertainty source interact to determine the ultimate measure of model reliability in its application domain, termed *predictive capability*. To quantify the effect of these process-dependent sources of uncertainty, the model must, typically, be run many times. Many models however are complex enough to only allow a single or a handful of runs at any particular setting, necessitating the need for reduced-order models. These approximations of the full model, also called *surrogates*, come with their own surrogate modeling uncertainty. Together, these uncertainties are referred to here as *total uncertainty*. 


## The discrepancy between non-determinstic simulations and experiments

```{tip}
Validation is a process of determining the degree to which a (non-deterministic) model simulation is an accurate representation of the corresponding (imperfect) physical experiment.
```

- Both model predictions and empirical data may contain both aleatory and epistemic uncertainty.

- Effectively the discrepancy measure is between two uncertain data generating processes.

- The discrepancy measure indicated the model-form uncertainty

```{image} ../_static/area_metric_illustration_3.png
:alt: Illustration of the stochastic area metric
:class: bg-primary
:width: 400px
:align: center

Illustration of the stochastic area metric
```

```python
from pyuncertainnumber import area_metric

dist = pba.normal(4, 1) data_sample = dist.sample(5)
ecdf_ = pba.ECDF(data_sample)
area_metric(dist, ecdf_) 
0.33125451465728933
```

## Predictive capability

Predictive capability suggests the prediction, along with its uncertainty estimation, with respect to a scenario in the application domain that is likely to be beyond the validation domain. Such predictive uncertainty shall comprise all the possible sources of uncertainties during the computational pipeline, as mentioned above. Notably, some uncertainties are of aleatory nature while others are of epistemic nature. The mechanism of probability bounding is employed to aggregate the effects of all these uncertainties to produce a reliable prediction.

```{image} ../_static/pbox_layers_static.png
:alt: Predictive capability
:class: bg-primary
:width: 400px
:align: center

Predictive capability
```



## An open challenge of VVUQ on aerodynamics

An open challenge has been proposed to focuses on estimating the predictive capability of an aerodynamic analysis tool (XFOIL) given a set of synthetic experimental data in AIAA Sci-Tech Forum 2026 {cite:p}`cary2026summary`.

```{image} ../_static/aiaa_challenge_statement.png
:alt: Problem statement 
:class: bg-primary
:width: 400px
:align: center

Problem statement of the AIAA Second Uncertainty Quantification Challenge Problem for Aerodynamics
```

In response to this challenge, The UQ team at UoL presents a methodological framework {cite:p}`chen2026efficient` for tackling the Second Uncertainty Quantification Challenge Problem for Aerodynamics. The challenge requires assessments of validation and predictive capability for an aerodynamic tool given imperfect experimental measurements that are scarce and imprecise. We propose an efficient interval-based integrated approach that comprehensively accounts for multiple sources of uncertainties including experimental data uncertainties, inputs incertitude, model-form discrepancy, numerical approximation, surrogate model prediction, and, importantly, uncertainties related to extrapolated application domains. These uncertainties are tackled by efficient meta-modelling techniques involving interval predictor models, Bayesian optimization on the basis of Gaussian processes, and stochastic neural network models. Our results underscore the effectiveness and efficiency of the approach in achieving a practical balance between comprehensive uncertainty quantification and computational cost, demonstrating broad applicability for uncertainty-driven validation and assessment of predictive capability in engineering applications.



## References

```{bibliography}
:filter: docname in docnames
```