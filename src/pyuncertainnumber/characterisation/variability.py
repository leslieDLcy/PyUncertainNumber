import enum

class Variability(enum.Enum):
    point_estimate = 'point estimate'
    confidence = 'Confidence'


''' hint '''
# d = Door('open')
# print(d.value)
# d = Door('closed')
# print(d.value)
# d = Door('is not a valid Door')
# print(d.value)