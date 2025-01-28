import enum

class Uncertainty_types(enum.Enum):
    Certain = 'certain'
    Aleatory = 'aleatory'
    Epistemic = 'epistemic'
    Inferential = 'inferential'
    Design_uncertainty = 'design uncertainty'
    Vagueness = 'vagueness'
    Mixture = 'mixture'

''' hint '''
# d = Door('open')
# print(d.value)
# d = Door('closed')
# print(d.value)
# d = Door('is not a valid Door')
# print(d.value)