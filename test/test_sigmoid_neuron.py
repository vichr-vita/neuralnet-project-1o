import unittest
from neural_net.sigmoid_neuron import SigmoidNeuron
import random

class PerceptronTestCase(unittest.TestCase):

    def test_basic(self):
        p = SigmoidNeuron(
                        [random.choice([0, 1]) for n in range(3)],
                        threshold=4
                        )
