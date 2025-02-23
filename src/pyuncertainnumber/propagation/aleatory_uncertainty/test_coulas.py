# import openturns as ot

# x1 = ot.Uniform(1, 3)
# x2 = ot.Normal(0, 2)
# copula = ot.IndependentCopula()

# X = ot.ComposedDistribution([x1, x2], copula)


# z= X.getSample(5)

# print(z)


import matplotlib.pyplot as plt
from pycopula.copula import ArchimedeanCopula
from pycopula.visualization import concentrationFunction
from pycopula.simulation import simulate

# The Clayton copula
clayton = ArchimedeanCopula(dim=2, family='clayton')

# Sampling of size 1000 with Clayton copula
sim = simulate(clayton, 1000)

# Computing theoritical and empirical concentrations functions
downI, upI, tailDown, tailUp = concentrationFunction(sim)
ClaytonDown = [ clayton.concentrationDown(x) for x in downI ]
ClaytonUp = [ clayton.concentrationUp(x) for x in upI ]

# Plotting
plt.figure()
plt.plot(downI, tailDown, color='red', linewidth=3, label="Empirical concentration")
plt.plot(upI, tailUp, color='red', linewidth=3)
plt.plot(downI, ClaytonDown, color='blue', linewidth=1, label="Clayton concentration")
plt.plot(upI, ClaytonUp, color='blue', linewidth=1)
plt.plot([0.5, 0.5], [0, 1], color='gray', alpha=0.6, linestyle='--', linewidth=1)
plt.title("Lower-Upper tail dependence Coefficients")
plt.xlabel("Lower Tail      Upper Tail")
plt.legend(loc=0)

plt.show()