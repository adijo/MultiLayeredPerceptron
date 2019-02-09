import unittest
from layers import AbstractLayer, HiddenLayer, FinalLayer
from literals.activations import ActivationLiterals
from literals.initialization import InitLiterals
import numpy as np


class TestHiddenLayer(unittest.TestCase):
    def test_forward_pass(self):
        # GIVEN
        in_dim = 2
        out_dim = 4
        layer = HiddenLayer(in_dim, out_dim, ActivationLiterals.RELU, InitLiterals.NORMAL)

        input_matrix = np.array([[1, 2],
                                 [4, 5]])
        batch_size = input_matrix.shape[1]

        # WHEN
        transformed_matrix = layer.forward(input_matrix)

        # THEN
        self.assertEqual(input_matrix.shape[0], in_dim)
        self.assertEqual(transformed_matrix.shape, (out_dim, batch_size))

    def test_final_layer_backward_pass(self):
        # GIVEN
        in_dim = 2
        out_dim = 3

        layer = FinalLayer(in_dim, out_dim, ActivationLiterals.SOFTMAX, InitLiterals.NORMAL)
        input_matrix = np.array([[1, 2],
                                 [4, 5]])

        true_labels = np.array([[1, 0],
                                [0, 1],
                                [0, 0]])

        predictions = layer.forward(input_matrix)
        layer.backward(predictions, true_labels)
        layer.update(0.001)

    def test_hidden_layer_backward_pass(self):
        pass

    def test_final_layer_weight_update(self):
        pass

    def test_final_layer_bias_update(self):
        pass

    def test_hidden_layer_weight_update(self):
        pass

    def test_hidden_layer_bias_update(self):
        pass
