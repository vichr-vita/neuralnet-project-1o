import numpy as np




class NeuralNet:
    def __init__(self, layer_sizes: list) -> None:
        self.layer_sizes = layer_sizes
        self.biases = [np.zeros(x) for x in self.layer_sizes]
        self.weights = [np.zeros(el) for el in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]


    def initialize_weights(self, type=None):
        """
        initialize weights in the network to a random number
        TODO: choose the initialization method
        """
        self.weights = [np.random.randn(el[0], el[1]) * np.sqrt(2/el[0]) for el in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

# def activation(layer_no: int, w, b):
#     if layer_no == 0:
#         return 
#     else:
#         return activation(layer_no - 1)
