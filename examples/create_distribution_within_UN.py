from PyUncertainNumber.characterisation.uncertainNumber import UncertainNumber as UN

# Create a Gausssian distribution within the UN

c = UN(
    name="elastic modulus",
    symbol="E",
    units="Pa",
    essence="distribution",
    distribution_initialisation=["gaussian", [0, 1]],
)
