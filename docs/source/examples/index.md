# Examples

<!-- Register the notebooks with Sphinx -->
```{toctree}
:maxdepth: 1

examples/getting_started
examples/example_dependency_dev_purpose
examples/repeated_variable
examples/linguistic_approximation
examples/significant_digits
```

<!-- Show a grid of nice cards that link to each notebook -->
::::{grid} 1 2 2 3
:gutter: 2

<!-- :::{card} Getting started
:link: getting_started
:link-type: doc
:img-top: ../_static/illustration_get_started.png
:class-card: sd-text-center        # optional: center content
:class-img-top: card-img-square    # <-- add a class you’ll style
Getting started with `pyuncertainnumber`
::: -->

:::{card} Dependency structures in uncertainty analysis
:link: example_dependency_dev_purpose
:link-type: doc
:img-top: ../_static/distribution_dependency.png
:class-card: sd-text-center        # optional: center content
:class-img-top: card-img-square    # <-- add a class you’ll style
Random dependencies can be known, partially known, or unknown
:::

:::{card} Interval dependency and repeated variables
:link: repeated_variable
:link-type: doc
:img-top: ../_static/function_hint.png
:class-card: sd-text-center        # optional: center content
:class-img-top: card-img-square    # <-- add a class you’ll style
Repeated variable prblem in interval analysis
:::

:::{card} Interpret linguistic hedges
:link: linguistic_approximation
:link-type: doc
:img-top: ../_static/about_200.png
:class-card: sd-text-center        # optional: center content
:class-img-top: card-img-square    # <-- add a class you’ll style
Interpret the uncertainty indicated by linguistic hedges (e.g. "about 7")
:::

:::{card} Significance of significant digits
:link: significant_digits
:link-type: doc
:img-top: ../_static/illustration_sigdigits.png
:class-card: sd-text-center        # optional: center content
:class-img-top: card-img-square    # <-- add a class you’ll style
Explore the uncertainty indicated by significant digits of numbers
:::

::::