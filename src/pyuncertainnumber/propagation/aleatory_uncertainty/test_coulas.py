import openturns as ot

x1 = ot.Uniform(1, 3)
x2 = ot.Normal(0, 2)
copula = ot.IndependentCopula()

X = ot.ComposedDistribution([x1, x2], copula)


z= X.getSample(5)

print(z)