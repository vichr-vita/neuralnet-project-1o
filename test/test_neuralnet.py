import unittest
from neural_net.neuralnet import NeuralNet

class NeuralNetTestCase(unittest.TestCase):

    def test_instantiation(self):
        nn = NeuralNet(layer_sizes=[4, 5, 7, 1])