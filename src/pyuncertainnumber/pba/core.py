from abc import ABC, abstractmethod


class Joint(ABC):

    def __init__(self, copula, marginals: list):
        self.copula = copula
        self.marginals = marginals
