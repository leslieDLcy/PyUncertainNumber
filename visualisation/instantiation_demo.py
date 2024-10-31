
from PyUncertainNumber import UncertainNumber as UN

# vacuous instantiation
un = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa')

# Interval instantiation
un = UN(name='elas_modulus', 
            symbol='E', 
            units='Pa', 
            essence='interval', 
            bounds=[2,3]
            )

# distribution instantiation
un = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa', 
    essence='distribution', 
    distribution_parameters=['gaussian', (1,2)]
    )

# explicit parametric pbox instantiation
un = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa', 
    essence='pbox', 
    distribution_parameters=['gaussian', [(0,12),(1,4)]]
    )

un.display(style='band')


# implicit parametric pbox instantiation
un = UN(
    name='elas_modulus', 
    symbol='E', 
    units='Pa', 
    essence='distribution', 
    distribution_parameters=['gaussian', [(0,12),(1,4)]]
    )

un.display(style='band')