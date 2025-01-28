import enum

class Measurand(enum.Enum):
    count = 'count'
    tally = 'tally'
    unobservable_parameter = 'unobservable parameter'
    probability = 'probability'
    distribution = 'distribution'
    range_ = 'range'
    rank = 'rank'

''' hint '''
# d = Door('open')
# print(d.value)
# d = Door('closed')
# print(d.value)
# d = Door('is not a valid Door')
# print(d.value)