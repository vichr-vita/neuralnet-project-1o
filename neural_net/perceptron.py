import numpy as np

class Perceptron:
    def __init__(self, inputs: list, threshold = 2) -> None:
        self.threshold = threshold
        self.inputs = inputs
        self.weights = np.array([np.random.rand() for n in range(len(inputs))])
        self.xw_sum = np.sum(self.weights * self.inputs)
        self.output = 1 if self.xw_sum + self.threshold > 0 else 0
