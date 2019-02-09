from abc import ABC, abstractmethod
import numpy as np
from literals.initialization import InitLiterals


class AbstractInitialization(ABC):

    def __init__(self):
        pass

    @staticmethod
    def initialize_weights(in_dim, out_dim, method_name):
        initializations = {
            InitLiterals.ZERO: Zero(),
            InitLiterals.NORMAL: Normal(),
            InitLiterals.GLOROT: Glorot()
        }

        if method_name not in initializations:
            raise Exception("Invalid Initialization. %s is not supported." % method_name)

        return initializations[method_name].initialize(in_dim, out_dim)

    @abstractmethod
    def initialize(self, in_dim, out_dim):
        pass


class Zero(AbstractInitialization):
    def initialize(self, in_dim, out_dim):
        return np.zeros(shape=(out_dim, in_dim))


class Normal(AbstractInitialization):
    def initialize(self, in_dim, out_dim):
        return np.array([[np.random.standard_normal() for _ in range(in_dim)] for _ in range(out_dim)])


class Glorot(AbstractInitialization):
    def initialize(self, in_dim, out_dim):
        d = np.sqrt(6 / (in_dim + out_dim))
        return np.array([[np.random.uniform(-d, d) for _ in range(in_dim)] for _ in range(out_dim)])
