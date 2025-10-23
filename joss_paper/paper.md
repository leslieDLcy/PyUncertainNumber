---
title: 'PyUncertainNumber for uncertainty propagation with black-box models: beyond probabilistic arithmetic'
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

<!-- another title -->
<!-- title: 'PyUncertainNumber for uncertainty propagation: beyond probabilistic arithmetic' -->

# Summary

Scientific computations or simulations play a central role in quantifying the performance, reliability, and safety of complex engineered systems. However, these analyses are complicated by the various sources of uncertainties inherent in the computational pipeline. To ensure that complex engineered systems can be operated reliably and robustly even during rare and extreme environment conditions, a comprehensive uncertainty analysis is required. The analysis should be comprehensive in two senses: (i) all possible sources of uncertainty should be identified and represented using approaite mathematical constructs; (ii) The mixture of various types of uncertainties shuold be rigorously propagated through. 
`pyuncertainnumber` enables such comprehensive analysis, by computing guaranteed bounds on functions of uncertain variables, intrusively and non-intrusively, given only partial empirical knowledge.


<!-- given only partial knowledge of the input probability distributions and their dependencies. -->

<!-- Critical challenges include xxx, code accessbility, tools to conduct the analysis. -->
<!-- By xxx, `pyuncertainnumber` bla bla.. non-intrusively. -->


<!-- Challenges include xxx, code accessbility, tools to conduct the analysis. -->

<!-- To quantitatively
account for uncertainty is vital in performance, relibiability, and safety of high-consequence systems. However, the challenge xxx of . More expressive frameworks are proposed to manage uncertainties in an imprecise setting. It is desired to  -->


# Statement of need

A comprehensive uncertainty framework for scientific computation involves a mathematical model,
through which various input uncertainties are propagated to estimate the uncertainty of an unknown quantity of interest (QoI).
Real world complex systems (physical or engineered) of industrial significance typically involves input parameters subject to uncertainties of various nature [@oberkampf:2004]. These input uncertainties are commonly manifested as mixed uncertainties, e.g. probability boxes (p-boxes) which effectively represents a set of distributions, combining both the aleatory and epistemic uncertainty in one structure, or a mixture of uncertainties encompassing, for instance, a vector of inputs parameters of aleatory (e.g. probability distributions), epistemic (e.g. intervals), and mixed nature (e.g. probability boxes).

Probability bounds analysis [@ferson:2001] is one of the expressive frameworks proposed to manage uncertainties in an imprecise setting [@beer:2013].
Software packages have been developed to facilitate the calculations of uncertain quantities, such as interval arithmetic [@marco_2022_6205624] and probability arithemetic [@gray:2021; @gray:2022]. Collectively, they can be referred to as *uncertainty arithmetic* [@chen:2025] which straightforwardly computes the response provided the performance function.

While it has the potential to automatically compile non-deterministic subroutines via uncertain primitives, its usages face several challenges, one significant challenge is that code accessibility of the simulation model (e.g. finite element or computational fluid dynamics models) of the system is often not guaranteed and hence unable to proceed. This would largely restrict the adoption of mixed uncertainty calculations in real-world applications of computational engineering and physics.

<!-- Such need has been echoed in the engineering applications and also the NASA challenge. -->

<!-- besides known issues such as [dependency problems](https://pyuncertainnumber.readthedocs.io/en/latest/examples/repeated_variable.html) -->

`pyuncertainnumber` addresses this challenge by providing the non-intrusive capability designed to allow generic black-box models to be rigorously propagated with polymorphic uncertainty.
This capability significantly boosts versatility for scientific computations through interfacing with many engineering softwares.






# Interval propagation in a non-intrusive manner


Interval analysis [@moore:2009] features the advantages of providing rigorous enclosures of the solutions to problems, especially for engineering problems
subject to epistemic uncertainty, such as modelling system paramters due to lack-of-knowledge or characterising measurement incertitude.
Naive interval arithmetic typically faces difficulties such as the infamous [interval dependency](https://pyuncertainnumber.readthedocs.io/en/latest/examples/repeated_variable.html) issue. 
Though it may be mitigated through arithmetic rearrangements in some simple cases, it still can challenging for models of most complex systems. 
The critical issue remains the accessibility of code.

<!-- But naive interval arithmetic faces xxx problems, though xxx provides mathematical re-arrangements.  -->

Generally, the interval propagation problem can be cast as an optimisation problem where the minimum and maximum are sought via a function mapping.
The function, for example $f$ in \autoref{eq:intervalpropagation}, is not necessarily monotonic or linear and may well be a black-box deterministic model for a generic system. 

\begin{equation}\label{eq:intervalpropagation}
Y = f(I_{x1}, I_{x2}, ..., I_{xn})
\end{equation}

where $\mathbf{I} = [\mathbf{\underline{I}}, \mathbf{\overline{I}}] = [I_{x1}, I_{x2}, ..., I_{xn}]^\text{T}$ represents the vector of interval-valued inputs.
For black box models the optimisation can generally be solved via gradient-free optimisation techniques.

$$\underline{Y} = \min_{\underline{\mathbf{I}} \leq \mathbf{I} \leq \overline{\mathbf{I}} } [f(\mathbf{I})]; \ \overline{Y} = \max_{\underline{\mathbf{I}} \leq \mathbf{I} \leq \overline{\mathbf{I}} } [f(\mathbf{I})]$$


`pyuncertainnumber` provides a series of non-intrusive methodologies of varying applicability. It should be noted that there is generally a trade-off between 
applicability and computational efficiency. With more knowledge pertaining the characteristics of the underlying function, one can accordinly dispatch an efficient method.
For example, when monotonicity is known one can use the vertex method which requres $2^n$ model evaluations. Furthermore, the accuracy of these methods varies, and a common rule of thumb indicates that increasing the number of model evaluations generally leads to improved accuracy.
A summary of applicability is tabulated in \autoref{tab:ipmethods}, readers can refer to [@chen:2025] for additional details.

<!-- tabulate the interval results from the example -->
<!-- think twice. the middle row can be changed into a paragraph in the main text instead -->

<!-- Table: Several methods for interval propagation []{label="tab:ipmethods"} -->
Table: Supported methods for non-intrusive interval propagation.\label{tab:ipmethods}

| Method     | Endpoints    | Subinterval reconstitution | Cauchy-Deviate method           | Bayesian optimisation | Genetic algorithm |
|------------|--------------|----------------------------|---------------------------------|-----------------------|-------------------|
| Assumption | monotonicity | monotonicity in subinterlvas          | linearity and gradient required | No                    | No                |
| Example result     |  [13.0,148.0]            |   [13.0,148.0]                         |     [-11.7,100.67]                            |     [13.0,148.0]                  |   [13.0,147.8]                |

To better demonstrate the non-intrusive capability, two numerical examples, as displayed below in \autoref{fig:two_functions}, are provided where they are treated as black-box models. 
\autoref{tab:ipmethods} lists the response interval of $f_{b}([1,5], [7,13], [5,10])$ for respective methods.

<!-- The functional forms are displayed in \autoref{fig:dmc} and \autoref{fig:imc} respectively. -->

![Exampler functions as black-box models. (a) $f_{a}(x, y) = 100(x-y^2)^2 + (1 - x)^2$; (b) $f_{b}(x, y, z) = x^3 + y +z$\label{fig:two_functions}](photo_appending.png)



# Mixed uncertainty propagation for black-box models

Real world complex systems (physical or engineered) of industrial significance typically involves parameters subject to uncertainties of various nature [@oberkampf:2004]. It requires faithful characterisation of these uncertainties given the empirical information, and the approaach to rigorously progate them. Due to the fact that empirical information is often sparse or scarce or conflicting, even the uncertainty characterisation for one parameter could be of mixed nature, for example one may be confident about the distributional family but uncertain about its shape parameters, or when there exists multiple expert opiontion of different credibility regarding its elicitation. 
Commonly, real systems expect a high-dimensional input which effectively represents a mixture of aleatory, epistemic, and mixed uncertainties, as symbolfied below:

<!-- Imprecise world bla bla. After faithful characterisation, the ability to propagate is the key in many critical engineering applications.  -->

\begin{equation}
Y = f(\mathbf{u}; C)
\end{equation}

where $\mathbf{u} \in \mathbb{R}^{n}$ denotes the collection of $n$ uncertain inputs and $C$ denotes intervariable dependency structure.


When both aleatory and epistemic uncertainties are present in $\mathbf{u}$, a *nested (double) Monte Carlo* approach can be used for determininsitc models without confounding the two distinct types of uncertainty.
As illustrated in \autoref{fig:dmc}, Latin-hypercube samples are first drawn from the epistemic interval, conditioned on which aleatory samples are drawn from the aleatoric probability distributions. 
Propagate these samples, which are visually denoted as rug ticks alongside the abscissa, through the computational model results in an ensemble of CDF (cumulative distribution function) of the QoI whereby a final p-box is obtained as the envelope. 
Each CDF (orange color) correponds to an epistemic sample. 

![Workflow of the Double Monte Carlo method.\label{fig:dmc}](dmc_flowchart.png)


To scale to a more realistic setting, \autoref{fig:imc} illustrates the workflow of *interval Monte Carlo* method where a mixture of aleatory, epistemic, and mixed uncertainty parameters are present, and a certain copula is specified denoting the dependency structure. Correlated samples in the uniform space from the copula, visually denoted as rug ticks alongside the probability axis, are converted to physical space through alpha-cuts. Interval propagation (see [the last section](#interval-propagation-in-a-non-intrusive-manner)) then does the heavy lifting in which scalar values can be considered as degenerate intervals. As a result, the response QoI in \autoref{fig:imc} is then obtained as a p-box shown in gray. In contrast, a pinched response, obtained from propagating pinched input variables (e.g. a p-box is pinched into a distribution and an interval is pinched into a scalar), is also shown as a comparison. Importantly, the pinched result being enclosed  in the p-box manifests a critical feature of probability bounds analysis which yields results that are guaranteed to encolse all possible distributions of the ouput so long as the input p-boxes were all sure to enclose their respective distributions.

![Workflow of the Interval Monte Carlo method. The first row shows the marginal of the inputs and the second row shows the bivariate copula density. \label{fig:imc}](imc_diagram2.png)




# Conclusion

<!-- reiterate the significance of our developments  !!! -->

It is evident that many computational tasks in engineering and physics rely on complex numerical methods in which the model functions are black boxes.
This makes uncertainty arithmetic difficult to cope due to the code accessiability of models.
By providing methods supporting generic black box models, `pyuncertainnumber` enables rigorous uncertainty analysis for real-world scenarios of mixed uncertainties and partial knowledge.
This non-intrusive capability allows `pyuncertainnumber` to interface with multiple software environments to provide extensive compatibility with engineering applications, thereby its applicability across diverse computational workflows.

<!-- We interface with many softwares.
Significance: this provides compatability as interfacing with many engineering applications.
boost its usage.

Enriched sampling methods bla bla ... -->
<!-- # Propagation of p-boxes via surrogate models -->



# Acknowledgements

The work leading to these results received funding through the UK project Development of Advanced Wing Solutions 2 (DAWS2). The DAWS2 project is supported by the Aerospace Technology Institute (ATI) Programme, a joint government and industry investment to maintain and grow the UK’s competitive position in civil aerospace design and manufacture. The programme, delivered through a partnership between ATI, Department for Business and Trade (DBT) and Innovate UK, addresses technology, capability and supply chain challenges.

# References