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
        return z that has been transformed through sigmoid func

        loses precision for values close to 0 and 1
        """
        return np.log(sig / (1 - sig))

    @staticmethod
    def sigmoid_prime(z):
        """
        return first derivative of the sigmoid function with respect to z
        """
        return NeuralNet.sigmoid(z)*(1-NeuralNet.sigmoid(z))

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
        save the state of activations and corresponding zs
        """
        if self.layer_sizes[0] != len(input_vec):
            raise NeuralNetException(f'input vector size ({len(input_vec)}) does not match input layer size ({self.layer_sizes[0]})')

        a = input_vec
        self.a_list = [a]
        self.z_list = []
        for w, b in zip(self.weights, self.biases):
            z = NeuralNet.z_func(a, w, b)
            self.z_list.append(z)
            a = NeuralNet.sigmoid(z)
            self.a_list.append(a)
        return a # final output layer

    @staticmethod
    def cost_feed(output_vec: np.ndarray, desired_output_vec: np.ndarray) -> float:
        """
        return a the cost of one feed

        output_vec: output of the network
        desired_output_vec: correct output

        shapes of the vectors are to be the same
        """
        return np.sum(np.power(output_vec - desired_output_vec, 2))

    def get_cost_gradient(self, input_vec: np.ndarray, desired_output_vec: np.ndarray):
        w_grad = [np.zeros(w.shape) for w in self.weights]
        b_grad = [np.zeros(b.shape) for b in self.biases]

        a = input_vec
        # a_list = [a]
        # z_list = []
        # for w, b in zip(self.weights, self.biases):
        #     z = NeuralNet.z_func(a, w, b)
        #     z_list.append(z)
        #     a = NeuralNet.sigmoid(z)
        #     a_list.append(a)

        self.feed_forward(a)
        
        delta = self.cost_partial_derivative_to_a(self.a_list[-1], desired_output_vec) * NeuralNet.sigmoid_prime(self.z_list[-1])
        w_grad[-1] = np.dot(delta[..., None], self.a_list[-2][None, ...])
        b_grad[-1] = delta

        for i in range(2, len(self.layer_sizes)):
            z = self.z_list[-i]
            delta = np.dot(self.weights[-i+1].T, delta) * NeuralNet.sigmoid_prime(z)
            w_grad[-i] = np.dot(delta[..., None], self.a_list[-i-1][None, ...])
            b_grad[-i] = delta
        return (w_grad, b_grad)

    
    @staticmethod
    def cost_partial_derivative_to_a(output_vec, desired_output_vec):
        return 2*(output_vec - desired_output_vec)
