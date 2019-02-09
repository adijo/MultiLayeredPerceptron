from activations import AbstractActivation, ReLU, Softmax
from literals.activations import ActivationLiterals
import unittest
import numpy as np
from nose.tools import raises


class TestActivation(unittest.TestCase):
    def test_relu_instantiation(self):
        # Given
        relu = AbstractActivation.from_activation_name(ActivationLiterals.RELU)

        # Then
        self.assertIsInstance(relu, ReLU)

    def test_softmax_instantiation(self):

        # Given
        softmax = AbstractActivation.from_activation_name(ActivationLiterals.SOFTMAX)

        # Then
        self.assertIsInstance(softmax, Softmax)

    @raises(Exception)
    def test_invalid_activation(self):
        # Given
        AbstractActivation.from_activation_name("invalid")

    def test_relu_apply_activation(self):

        # Given
        pre_activation = np.array([[1], [0], [-4]])
        relu = ReLU()

        # When
        activation = relu.apply_activation(pre_activation)

        # Then
        expected = np.array([[1], [0], [0]])
        self.assertTrue(np.array_equal(activation, expected))

    def test_relu_calculate_gradient(self):
        # Given
        pre_activation = np.array([[1, 2], [3, -3], [0, -2]])
        relu = ReLU()

        # When
        grad = relu.calculate_gradient(pre_activation, None)

        # Then
        expected = np.array([[1, 1], [1, 0], [0, 0]])
        self.assertTrue(np.array_equal(grad, expected))

    def test_softmax_apply_activation(self):
        # Given
        pre_activation = np.array([[1, 2, 3, 6],
                                  [2, 4, 5, 6],
                                  [3, 8, 7, 6]])
        softmax = Softmax()

        # When
        activation = softmax.apply_activation(pre_activation)

        # Then
        sum_of_columns = 4.0
        self.assertTrue(np.isclose(sum_of_columns, np.sum(np.sum(activation, axis=0)), 1e-3))

    def test_softmax_calculate_gradient(self):
        # Given
        pre_activation = np.array([[1, 2, 3, 6],
                                  [2, 4, 5, 6],
                                  [3, 8, 7, 6]])
        target = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]])
        softmax = Softmax()

        # When
        activation = softmax.apply_activation(pre_activation)
        grad = softmax.calculate_gradient(activation, target)
