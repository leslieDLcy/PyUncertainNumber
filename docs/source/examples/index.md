# Examples

<!-- Register categories with Sphinx; notebooks are included in each category page -->
```{toctree}
:maxdepth: 1
:titlesonly:
:hidden:

propagation/index
calibration/index
characterisation/index
```

## Propagation

::::{grid} 1 2 2 3
:gutter: 2

:::{card} Aleatory propagation
:link: propagation/aleatory_propagation_demo
:link-type: doc
:img-top: ../_static/aleatory_propagation_demo.png
Propagate random variables via performance functions.
:::

:::{card} Interval propagation
:link: propagation/interval_propagation_demo
:link-type: doc
:img-top: ../_static/interval_propagation_demo.png
Epistemic/interval bounds, intrusive and non-intrusive.
:::

:::{card} Mixed uncertainty
:link: propagation/mix_uncertainty_propagation_demo
:link-type: doc
:img-top: ../_static/illustration_get_started.png
Hybrid aleatory–epistemic propagation examples.
:::

::::

## Calibration

::::{grid} 1 2 2 3
:gutter: 2

:::{card} Transitional MCMC (2-DOF)
:link: calibration/2dof_tmcmc_demo
:link-type: doc
:img-top: ../_static/tmcmc_2dof.png
Posterior inference of stiffness parameters with TMCMC.
:::

:::{card} KNN calibration
:link: calibration/KNN_calibrator_demo
:link-type: doc
:img-top: ../_static/KNN_calibration_demo.png
Data-driven likelihood-free calibration technique.
:::

:::{card} Data peeling algorithm
:link: calibration/Data_peeling_demo
:link-type: doc
:img-top: ../_static/nested_ds.png
Data peeling algorithm on a banana-shaped data generating process.
:::

::::

## Characterisation

::::{grid} 1 2 2 3
:gutter: 2

:::{card} Work with intervals
:link: characterisation/work_with_interval
:link-type: doc
:img-top: ../_static/interval_illustration.png
Basics of interval arithmetic and usage patterns.
:::

:::{card} Dependency structures
:link: characterisation/example_dependency_dev_purpose
:link-type: doc
:img-top: ../_static/distribution_dependency.png
Illustrates dependence assumptions in uncertainty analysis.
:::

:::{card} Repeated variables
:link: characterisation/repeated_variable
:link-type: doc
:img-top: ../_static/function_hint.png
Handling repeated-variable issues in interval analysis.
:::

:::{card} Linguistic approximation
:link: characterisation/linguistic_approximation
:link-type: doc
:img-top: ../_static/about_200.png
Interpret uncertainty expressed with linguistic hedges.
:::

:::{card} Significant digits
:link: characterisation/significant_digits
:link-type: doc
:img-top: ../_static/illustration_sigdigits.png
Explore information carried by significant digits.
:::

:::{card} Characterise what you know
:link: characterisation/characterise_what_you_know
:link-type: doc
:img-top: ../_static/free_pbox_constraint_demo.png
Empirical constraints for characterising uncertain numbers.
:::

::::
::::