import enum

""" hint 

Only a pre-set of values are allowed for certain fields in the UncertainNumber class.
"""
# d = Door('open')
# print(d.value)
# d = Door('closed')
# print(d.value)
# d = Door('is not a valid Door')
# print(d.value)

""" ancillary informtion about the uncertain number"""


class AncillaryUncertainty(enum.Enum):
    """
    Base class for all enums that describe different aspects of uncertainty.

    We aim to organise the ancillary information into a list of fields to be
    stored with the quantitative details of the uncertain number rather
    than separately as comments or narrative statements in other files.


    These fields are intended to represent more-or-less orthogonal descriptors that together
    fully characterise what we need to know about the quantity in order to use it in calculations.
    The collection of these fields should be rich enough to account for all relevant and useful
    information, but it is critical that it not seem daunting or burdensome to users. Users are
    never required to enter anything into any of these fields. Numbers that lack substantive
    justifications are indicated with visual cues but, wherever possible, they are not otherwise
    delayed, penalised or restricted in use. Some of the entries for these various fields can be
    populated automatically by computer, and several could be, especially for numbers that arise
    from computing with previously defined numbers.

    The values for the descriptor fields can be given as arbitrary strings with arbitrary formatting
    or by links to other resources. The possible detail for any field is unlimited. For convenience,
    however, suggested entries for many fields can be selected from comprehensive (and
    customisable) dropdown menus.

    Subclasses:
        - Measurand: What is being measured or described (e.g., count, probability, distribution).
        - Ensemble: The collection or grouping over which variability or uncertainty is considered
          (e.g., repeated measurements, flights, households).
        - Variability: The way uncertainty is expressed or summarized
          (e.g., point estimate, confidence).

    """

    pass


# * ----------------------- measurand field


class Measurand(AncillaryUncertainty):
    """What is being measured or described (e.g., count, probability, distribution)."""

    count = "count"
    tally = "tally"
    unobservable_parameter = "unobservable parameter"
    probability = "probability"
    distribution = "distribution"
    range_ = "range"
    rank = "rank"


class Ensemble(AncillaryUncertainty):
    """The collection or grouping over which variability or uncertainty is considered
    (e.g., repeated measurements, flights, households)."""

    repeated_measurements = "repeated measurements"
    flights = "flights"
    pressurisations = "pressurisations"
    temporal_steps = "temporal steps"
    spatial_sites = "spatial sites"
    manufactured_components = "manufactured components"
    customers = "customers"
    people = "people"
    households = "households"
    particular_population = "particular population"


class Variability(AncillaryUncertainty):
    """The way uncertainty is expressed or summarized (e.g., point estimate, confidence)."""

    point_estimate = "point estimate"
    confidence = "confidence"


class Uncertainty_types(AncillaryUncertainty):
    """The type of uncertainty affecting the uncertain number."""

    Certain = "certain"
    Aleatory = "aleatory"
    Epistemic = "epistemic"
    Inferential = "inferential"
    Design_uncertainty = "design uncertainty"
    Vagueness = "vagueness"
    Mixture = "mixture"
