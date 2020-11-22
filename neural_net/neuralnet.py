import numpy as np
import random
import pickle

class NeuralNetException(Exception):
    pass

class NeuralNet:

    def __init__(self, layer_sizes: list, initialize=True) -> None:
        self.layer_sizes = layer_sizes
        self.biases = [np.zeros(x) for x in self.layer_sizes[1:]] # input layer doesn't have a bias
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

    def dump(self, path):
        """
        save the network to file
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        load the network from file
        """
        with open(path, 'rb') as f:
            return pickle.load(f)


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
        # TODO might not use
        return np.sum(np.power(output_vec - desired_output_vec, 2))


    def get_cost_gradient(self, input_vec: np.ndarray, desired_output_vec: np.ndarray) -> tuple:
        """
        backpropagation
        return dict of weights and biases gradients and MSE for one forward feed
        """
        w_grad = [np.zeros(w.shape) for w in self.weights]
        b_grad = [np.zeros(b.shape) for b in self.biases]

        a = input_vec

        self.feed_forward(a)
        
        delta = self.cost_partial_derivative_to_a(self.a_list[-1], desired_output_vec) * NeuralNet.sigmoid_prime(self.z_list[-1])
        w_grad[-1] = np.dot(delta[..., None], self.a_list[-2][None, ...])
        b_grad[-1] = delta

        for i in range(2, len(self.layer_sizes)):
            z = self.z_list[-i]
            delta = np.dot(self.weights[-i+1].T, delta) * NeuralNet.sigmoid_prime(z)
            w_grad[-i] = np.dot(delta[..., None], self.a_list[-i-1][None, ...])
            b_grad[-i] = delta
        return {
                'w grad': w_grad,
                'b grad': b_grad,
                'mse': NeuralNet.cost_feed(self.a_list[-1], desired_output_vec)
                }


    @staticmethod
    def cost_partial_derivative_to_a(output_vec, desired_output_vec):
        return 2*(output_vec - desired_output_vec)


    def update_weights_and_biases(self, batch, learning_rate: float) -> None:
        """
        calculate gradient of weights and biases for each input in a batch and add the total average gradient
        scaled by learning rate to the current weights and biases of the network

        return average MSE for this batch

        batch: list:    [(input, desired_output),
                        (input, desired_output)
                        .
                        .
                        .
                        (input, desired_output)]
        """
        # TODO: check if inputs and outputs are the same length and are vectors

        if not 0 <= learning_rate <= 1:
            raise NeuralNetException(f'learing rate {learning_rate} should be in [0, 1]')
        

        w_grad = [np.zeros(w.shape) for w in self.weights]
        b_grad = [np.zeros(b.shape) for b in self.biases]

        # calculate desired changes for each feed
        mse = 0.00
        for i, d_o in batch:
            # check if i and d_o are both vectors of the same length
            if i.shape != d_o.shape:
                raise NeuralNetException(f'{i.shape} is not the same as {d_o.shape}')
            cost_delta_grad_dict = self.get_cost_gradient(i, d_o)
            w_grad = [gw+dgw for gw, dgw in zip(w_grad, cost_delta_grad_dict['w grad'])]
            b_grad = [gb+dgb for gb, dgb in zip(b_grad, cost_delta_grad_dict['b grad'])]
            mse += cost_delta_grad_dict['mse']
        
        # update the network len(inputs) == len(desired_outputs)
        self.weights = [w-(learning_rate/len(batch))*nw for w, nw in zip(self.weights, w_grad)]
        self.biases = [b-(learning_rate/len(batch))*nb for b, nb in zip(self.biases, b_grad)]
        return mse/len(batch)

    def learn(self, labeled_training_dataset: list, no_epochs: int, mini_batch_size: int, learning_rate: float):
        """
        can yield intermediate state of learning for testing/serializing
        """
        for e in range(no_epochs):
            random.shuffle(labeled_training_dataset)
            mini_batches = [labeled_training_dataset[i:i+mini_batch_size] for i in range(0, len(labeled_training_dataset), mini_batch_size)]
            mse = 0.00
            for batch in mini_batches:
                mse += self.update_weights_and_biases(batch, learning_rate)
            mse = mse/len(mini_batches)
            yield {
                'epoch': e,
                'mse': mse,
                'state': self
            }
