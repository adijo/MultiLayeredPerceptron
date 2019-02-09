import unittest
from data import MNISTDataset


class TestMNISTDataset(unittest.TestCase):
    def test_load_data(self):
        # GIVEN
        batch_size = 2
        mnist_data_size = 60000
        mnist_input_dim = 784

        # WHEN
        mnist = MNISTDataset(batch_size)

        # THEN
        self.assertEqual(len(mnist), mnist_data_size)
        self.assertEqual(mnist.input_dimension, mnist_input_dim)
