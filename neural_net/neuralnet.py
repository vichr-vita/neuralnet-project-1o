import numpy as np
import random
import pickle
import neural_net.activation_functions as af


class NeuralNetException(Exception):
    pass


class NeuralNet:

    def __init__(self, layer_sizes: list, a_functions: list, a_functions_prime: list, initialize=True) -> None:
        """
        initalize neural network
        """
        self.layer_sizes = layer_sizes
        # input layer doesn't have a bias
        self.biases = [np.zeros(x) for x in self.layer_sizes[1:]]
        self.weights = [np.zeros(el) for el in zip(
            self.layer_sizes[1:], self.layer_sizes[:-1])]
        if len(a_functions) != len(self.layer_sizes) - 1 != len(a_functions_prime):
            raise NeuralNetException(
                f'layer sizes length ({len(layer_sizes)}) != ac func list length ({len(a_functions)}, {len(a_functions_prime)})')
        self.a_functions = a_functions
        self.a_functions_prime = a_functions_prime
        self.mse_avg = 0

        if initialize:
            self._initialize_weights()
            self._initialize_biases()

    def _initialize_weights(self, type=None):
        """
        initialize weights in the network to a random number
        TODO: choose the initialization method in the future
        """
        self.weights = [np.random.randn(*w.shape) for w in self.weights]

    def _initialize_biases(self, type=None):
        """
        initialize biases in the network to a random number
        TODO: choose the initialization method in the future
        """
        self.biases = [np.random.randn(*b.shape) for b in self.biases]

    @staticmethod
    def dump(obj, path):
        """
        save the network to file

        TODO: save functions
        """
        raise DeprecationWarning('load is currently deprecated')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path):
        """
        load the network from file

        TODO: save functions
        """
        raise DeprecationWarning('load is currently deprecated')
        with open(path, 'rb') as f:
            return pickle.load(f)

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
            raise NeuralNetException(
                f'input vector size ({len(input_vec)}) does not match input layer size ({self.layer_sizes[0]})')

        a = input_vec
        self.a_list = [a]
        self.z_list = []
        for w, b, func in zip(self.weights, self.biases, self.a_functions):
            z = NeuralNet.z_func(a, w, b)
            self.z_list.append(z)
            a = func(z)
            self.a_list.append(a)
        # return self.a_list[-1] # final output layer
        return {
            'a': self.a_list[-1],
            'z': self.z_list[-1]
        }

    @staticmethod
    def cost_feed(output_vec: np.ndarray, desired_output_vec: np.ndarray) -> float:
        """
        return a the cost of one feed: mse

        output_vec: output of the network
        desired_output_vec: correct output

        shapes of the vectors are to be the same
        """
        return np.sum(np.power(output_vec - desired_output_vec, 2))

    def _get_cost_gradient(self, input_vec: np.ndarray, desired_output_vec: np.ndarray) -> tuple:
        """
        backpropagation
        return dict of weights and biases gradients and mse for one forward feed
        """
        w_grad = [np.zeros(w.shape) for w in self.weights]
        b_grad = [np.zeros(b.shape) for b in self.biases]

        a = input_vec
        activated_desired_ov = self.a_functions[-1](desired_output_vec)

        self.feed_forward(a)

        delta = self.cost_partial_derivative_to_a(
            self.a_list[-1], activated_desired_ov) * self.a_functions_prime[-1](self.z_list[-1])
        w_grad[-1] = np.dot(delta[..., None], self.a_list[-2][None, ...])
        b_grad[-1] = delta

        for i in range(2, len(self.layer_sizes)):
            z = self.z_list[-i]
            delta = np.dot(self.weights[-i+1].T, delta) * \
                self.a_functions_prime[-i](z)
            w_grad[-i] = np.dot(delta[..., None], self.a_list[-i-1][None, ...])
            b_grad[-i] = delta
        return {
            'w grad': w_grad,
            'b grad': b_grad,
            'mse': NeuralNet.cost_feed(self.a_list[-1], activated_desired_ov)
        }

    @staticmethod
    def cost_partial_derivative_to_a(output_vec, desired_output_vec):
        return 2*(output_vec - desired_output_vec)

    def _update_weights_and_biases(self, mini_batch, learning_rate: float) -> float:
        """
        calculate gradient of weights and biases for each input in a batch and add the total average gradient
        scaled by learning rate to the current weights and biases of the network

        return average mse for this batch

        batch: list:    [(input, desired_output),
                        (input, desired_output)
                        .
                        .
                        .
                        (input, desired_output)]
        """
        # TODO: check if inputs and outputs are the same length and are vectors

        w_grad = [np.zeros(w.shape) for w in self.weights]
        b_grad = [np.zeros(b.shape) for b in self.biases]

        # calculate desired changes for each feed
        mse = 0.00
        for i, d_o in mini_batch:
            # check if i and d_o are both vectors of the same length
            if i.shape != d_o.shape:
                raise NeuralNetException(
                    f'{i.shape} is not the same as {d_o.shape}')
            cost_delta_grad_dict = self._get_cost_gradient(i, d_o)
            w_grad = [gw+dgw for gw,
                      dgw in zip(w_grad, cost_delta_grad_dict['w grad'])]
            b_grad = [gb+dgb for gb,
                      dgb in zip(b_grad, cost_delta_grad_dict['b grad'])]
            mse += cost_delta_grad_dict['mse']

        self.weights = [w-(learning_rate/len(mini_batch)) *
                        nw for w, nw in zip(self.weights, w_grad)]
        self.biases = [b-(learning_rate/len(mini_batch)) *
                       nb for b, nb in zip(self.biases, b_grad)]
        return mse/len(mini_batch)

    def gradient_descent(self, labeled_training_dataset: list, no_epochs: int, mini_batch_size: int, learning_rate: float) -> dict:
        """
        can yield intermediate state of learning for testing/serializing
        last mse
        """
        len_training_data = len(labeled_training_dataset)
        for e in range(no_epochs):
            random.shuffle(labeled_training_dataset)
            mini_batches = [labeled_training_dataset[i:i+mini_batch_size]
                            for i in range(0, len_training_data, mini_batch_size)]
            mse = 0.00
            for mini_batch in mini_batches:
                mse = self._update_weights_and_biases(
                    mini_batch, learning_rate)
            yield {
                'epoch': e,
                'mse': mse,
                'state': self
            }

    def gradient_descent_testdata(self, labeled_training_dataset: list, labeled_test_dataset: list, no_epochs: int, mini_batch_size: int, learning_rate: float) -> dict:
        """
        can yield intermediate state of learning for testing/serializing
        train_mse: last mse of learning epoch
        test_mse: average mse for test feed
        """
        len_training_data = len(labeled_training_dataset)
        len_test_data = len(labeled_test_dataset)
        for e in range(no_epochs):
            random.shuffle(labeled_training_dataset)
            mini_batches = [labeled_training_dataset[i:i+mini_batch_size]
                            for i in range(0, len_training_data, mini_batch_size)]
            # train 
            train_mse = 0.00 # TODO: this metric might not be necessary anymore, delete?
            for mini_batch in mini_batches:
                train_mse = self._update_weights_and_biases(
                    mini_batch, learning_rate)


            # evaluate
            test_mse = 0.00
            for i, do in labeled_test_dataset:
                test_mse += self.feed_forward_performance(i, self.a_functions[-1](do))
            yield {
                'epoch': e,
                'train mse': train_mse,
                'test mse': test_mse/len(labeled_test_dataset),
                'state': self
            }

    def learn(self, labeled_training_dataset: list, no_epochs: int, mini_batch_size: int, learning_rate: float) -> None:
        """
        contained algorithm wrapping stochastic gradient descent
        """
        print('LEARN')
        for epoch_dict in self.gradient_descent(labeled_training_dataset, no_epochs, mini_batch_size, learning_rate):
            print('EPOCH:\t{}\tmse: {}'.format(
                epoch_dict['epoch'] + 1, epoch_dict['mse']), end='\r')
            self.last_epoch = epoch_dict['epoch']
            self.mse_avg += epoch_dict['mse']
        self.mse_avg = self.mse_avg/no_epochs
        print()
        print('DONE')

    def feed_forward_performance(self, input_vec: np.ndarray, desired_output_vec: np.ndarray) -> float:
        """
        return mse (TODO: implement for arbitrary error function) of one forward feed
        """
        output_vec = self.feed_forward(input_vec)['a']
        return NeuralNet.cost_feed(output_vec, desired_output_vec)



    @staticmethod 
    def ratio_list_split(l_split: list, ratio: float) -> tuple:
        """
        splits the dataset for training and evaluation
        """
        if not 0 <= ratio <= 1:
            raise ValueError('must be 0 <= ratio <= 1 is:', ratio)
        
        return (l_split[:int(len(l_split) * ratio)], l_split[int(len(l_split) * ratio):])


def normalize(z: np.ndarray) -> np.ndarray:
    return z * 1.0/z.max()
