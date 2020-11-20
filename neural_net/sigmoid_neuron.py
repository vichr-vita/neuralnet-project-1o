import numpy as np


class SigmoidNeuron:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    def __init__(self, inputs: list, threshold = 2) -> None:
        self.threshold = threshold
        self.inputs = inputs
        self.weights = np.array([np.random.rand() for n in range(len(inputs))])
        self.xw_sum = np.sum(self.weights * self.inputs)
        self.output = SigmoidNeuron.sigmoid(self.xw_sum)