from abc import ABC, abstractmethod
import numpy as np
from literals.activations import ActivationLiterals


class AbstractActivation(ABC):

    @staticmethod
    def from_activation_name(activation_name):
        activations = {
            ActivationLiterals.RELU: ReLU(),
            ActivationLiterals.SOFTMAX: Softmax()
        }
        if activation_name not in activations:
            raise Exception("Unsupported activation. %s is not supported" % activation_name)
        return activations[activation_name]

    @abstractmethod
    def apply_activation(self, x):
        pass

    @abstractmethod
    def calculate_gradient(self, x, y):
        pass


class ReLU(AbstractActivation):
    def apply_activation(self, data):
        return np.maximum(0, data)

    def calculate_gradient(self, data, target):
        return np.greater(data, 0).astype(int)


class Softmax(AbstractActivation):
    def apply_activation(self, data):
        data_ = data - np.max(data, axis=0)
        num = np.exp(data_)
        return num / (1e-10 + np.sum(num, axis=0))

    def calculate_gradient(self, predictions, target):
        assert(predictions.shape == target.shape)
        return predictions - target

