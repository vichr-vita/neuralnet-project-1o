import numpy as np
from numpy.core.fromnumeric import transpose

class NeuralNetException(Exception):
    pass

class NeuralNet:
    def __init__(self, layer_sizes: list, initialize=True) -> None:
        self.layer_sizes = layer_sizes
        self.biases = [np.zeros(x) for x in self.layer_sizes[1:]] # input layer doesn't have a bias

        # weight of Layer n is indexed as self.weights[n-1][to neuron of L n][from neuron of L n-1]
        self.weights = [np.zeros(el) for el in zip(self.layer_sizes[1:], self.layer_sizes[:-1])]

        if initialize:
            self.initialize_weights()
            self.initialize_biases()

    def initialize_weights(self, type=None):
        """
        initialize weights in the network to a random number
        TODO: choose the initialization method in the future
        """
        self.weights = [np.random.randn(*w.shape) for w in self.weights]

    def initialize_biases(self, type=None):
        """
        initialize biases in the network to a random number
        TODO: choose the initialization method in the future
        """
        self.biases = [np.random.randn(*b.shape) for b in self.biases]


    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        transform z to activations for layer l
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_inv(sig: np.ndarray) -> np.ndarray:
        """
        returns z that has been transformed through sigmoid func

        loses precision for values close to 0 and 1
        """

    @staticmethod
    def z_func(a: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        return vector z for layer l 
        it is importand to pass a as an ndarray even for one input neuron

        w(l-1) b(l)

        give a and b as row vectors
        """
        if type(a) is not np.ndarray:
            raise TypeError(type(a), 'is not', np.ndarray)
        return np.dot(w, a) + b

    def feed_forward(self, input_vec: np.ndarray) -> np.ndarray:
        """
        return a result as an output vector of shape (self.layer_sizes[-1],)
        """
        if self.layer_sizes[0] != len(input_vec):
            raise NeuralNetException(f'input vector size ({len(input_vec)}) does not match input layer size ({self.layer_sizes[0]})')

        a = input_vec
        for w, b in zip(self.weights, self.biases):
            a = NeuralNet.sigmoid(NeuralNet.z_func(a, w, b))
        return a

    @staticmethod
    def cost_func(output_vec: np.ndarray, desired_output_vec: np.ndarray) -> float:
        """
        return a value of the cost function

        output_vec: output of the network
        desired_output_vec: correct output

        shapes of the vectors are to be the same
        """
        return np.sum(np.power(output_vec - desired_output_vec, 2))