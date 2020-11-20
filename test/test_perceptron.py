import unittest
from neural_net.perceptron import Perceptron
import random

class PerceptronTestCase(unittest.TestCase):

    def test_basic(self):
        p = Perceptron(
                        [random.choice([0, 1]) for n in range(50)],
                        threshold=15
                        )
        self.assertTrue(0 <= p.output <= 1)
