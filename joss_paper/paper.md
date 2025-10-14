---
title: 'PyUncertainNumber for uncertainty propagation: more than just probability arithmetic'
tags:
  - Python
  - uncertainty propagation
  - imprecise probability
  - probability bound analysis 
authors:
  - name: Yu Chen
    orcid: 0000-0001-6617-2946
    corresponding: true
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Scott Ferson
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    corresponding: false
    affiliation: 1
  - name: Edoardo Patelli
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 2
affiliations:
 - name: Institute for Risk and Uncertainty, University of Liverpool, UK 
   index: 1
   ror: 00hx57361
 - name: Centre for Intelligent Infrastructure, University of Strathclyde 
   index: 2
date: 3 October 2025
bibliography: paper.bib
---

# Summary

The forces on stars, galaxies, and dark matter under external gravitational
fields lead to the dynamical evolution of structures in the universe. The orbits
of these bodies are therefore key to understanding the formation, history, and
future state of galaxies. The field of "galactic dynamics," which aims to model
the gravitating components of galaxies to study their structure and evolution,
is now well-established, commonly taught, and frequently used in astronomy.
Aside from toy problems and demonstrations, the majority of problems require
efficient numerical tools, many of which require the same base code (e.g., for
performing numerical orbit integration).

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the *Gaia* mission
[@gaia] by students and experts alike.


`pyuncertainnumber` enables rigorous uncertainty analysis for real-world situations 
of mixed uncertainties and partial knowledge. Aleatoric and epistemic uncertainties are 
recognised and treated appropriately in characterisation and propagation.

Uncertainty arithmetic is underpinned by probability bounds analysis. While it has the potential 
to automatically compile a non-deterministic subroutines via primitives such as intervals or uncertain numbers, 
its usages face several challenges.

Besides the issues of xx such as dependency problems, one notable challenge is that code accessibility is often not guaranteed. 
Also, the lack of capability one the main reasons restricting the adoption of xxx in practice.


`pyuncertainnumber` addresses that by enabling non-intrusive capability. How to work with black-box models? This capability significantly 
boost its versatility for scientific computations by interfacing with many engineering softwares.



# Interval propagation in a non-intrusive manner


Interval analysis has the advantages of providing rigorous enclosures of the solutions to problems, especially for engineering problems
subject to epistemic uncertainty, such as modelling system paramters due to lack-of-knowledge or characterising measurement incertitude.
It is evident that computational tasks requiring complex numerical solutions of intervals are non-intrusive (i.e. the source code is not accessiable).
Besides, it shoule be noted even for cystal boxes (i.e. source code is accessible), naive interval arithmetic still faces challlenges such as the infamous interval dependency issue. 
Though it may be mitigated through mathematical rearrangements in some cases, it will be challenging for most of the cases.
<!-- But naive interval arithmetic faces xxx problems, though xxx provides mathematical re-arrangements.  -->

Generally, the interval propagation problem can be cast as an optimisation problem where the minimum and maximum are sought via a function mapping.
The functio, for example $g$ in Eq.(xx), is not necessarily monotonic or linear and may well be a black-box model. Hence, for black box models the optimisation can 
only be solved via gradient-free optimisation techniques.

\begin{equation}
Y = g(I_{x1}, I_{x2}, ..., I_{xn})
\end{equation}


\begin{equation}
Y_min, Y_max
\end{equation}

where $I_{x1}, I_{x2}, ..., I_{xn}$ are intervals.

`pyuncertainnumber` provides a series of non-intrusive methodologies of varying applicability. It should be noted that there is generally a trade-off between 
applicability and efficiency. But with more knowledge about the characteristics of the underlying function, one can accordinly dispatch an efficient method.
For example, whem monotonicity is known one can use vertex methods which $2_n$.

<!-- tabulate the interval results from the example -->
<!-- think twice. the middle row can be changed into a paragraph in the main text instead -->

Table: Several methods for interval propagation []{label='ip_methods'}

| Method     | Endpoints    | Subinterval reconstitution | Cauthy-Deviate method           | Bayesian optimisation | Genetic algorithm |
|------------|--------------|----------------------------|---------------------------------|-----------------------|-------------------|
| Assumption | monotonicity | heavy computation          | linearity and gradient required | No                    | No                |
| Result     |              |                            |                                 |                       |                   |

As shown in \autoref{ip_methods}, tabulation of xxx given a black box model.
<!-- show the figure to indicate the ground-truth answer -->


# Mixed uncertainty propagation for black-box models


Most realistic situation bla bla. Imprecise world bla bla. After faithful characterisation, the ability 
to propagate is the key in many critical engineering applications. 

<!-- see the pbox propagation paper (iMC) and copy some texts herein -->

Dependency structures bla bla. It has been echoed in the engineering applications and also the NASA challenge.

Sampling methods play a significant role in xxx

Double Monte Carlo


Interval Monte Carlo...




# Propagation of p-boxes via surrogate models



<!-- 
# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text. -->

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Double Monte Carlo.\label{fig:dmc}](dmc_flowchart.png)
\autoref{fig:dmc} illustrates the *nested Monte Carlo* method.


![Interval Monte Carlo.\label{fig:imc}](imc_flowchart.png)
\autoref{fig:imc} illustrates the *interval Monte Carlo* method.



<!-- Figure sizes can be customized by adding an optional second parameter:
![Interval Monte Carlo.](imc_flowchart.png){ width=20% } -->

# Conclusion

Significance: this provides compatability as interfacing with many engineering applications.
boost its usage.

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References