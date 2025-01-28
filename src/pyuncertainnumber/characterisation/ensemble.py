import enum


class Ensemble(enum.Enum):
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

    # TODO user can edit at will...


""" hint """
# d = Door('open')
# print(d.value)
# d = Door('closed')
# print(d.value)
# d = Door('is not a valid Door')
# print(d.value)
