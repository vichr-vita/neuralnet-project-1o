from typing import Type
import numpy as np
import unittest
import warnings
import itertools
import json

from numpy.core.arrayprint import _leading_trailing
from neural_net.neuralnet import NeuralNet, NeuralNetException
import neural_net.activation_functions as af

class NeuralNetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass


    def test_basic(self):
        x = np.linspace(-4, 4, 1000)
        cosh = np.cosh(x)
        # make inputs the list of numpy arrays
        sig_x = [np.array([af.sigmoid(n)]) for n in x]
        sig_cosh = [np.array([af.sigmoid(n)]) for n in cosh]
        layer_sizes=[len(sig_x[0]), 3, 4, 1]
        nn = NeuralNet(layer_sizes=layer_sizes, a_functions=[af.sigmoid for n in layer_sizes[1:]], a_functions_prime=[af.sigmoid_prime for n in layer_sizes[1:]], initialize=False) # first layer created to match the input size
        self.assertTrue(len(nn.biases) == len(nn.layer_sizes) - 1)
        for b, s in zip(nn.biases, nn.layer_sizes[1:]):
            self.assertEqual(len(b), s)
        nn._initialize_weights()
        nn._initialize_biases()
        # print(nn.weights)

        wrong_type = 3
        self.assertRaises(TypeError, NeuralNet.z_func, wrong_type, nn.weights[0], nn.biases[0])
        
        layer_no = 1 # second layer
        training_data_index = np.random.randint(0, len(sig_x)) # random input from the data set

        z = NeuralNet.z_func(sig_x[training_data_index], nn.weights[layer_no - 1], nn.biases[layer_no - 1])
        self.assertEqual(len(z.shape), 1) # check if returns a vector, not a matrix
        self.assertEqual(z.shape[0], nn.layer_sizes[layer_no]) # check shape of the vector agains the layer
        self.assertTrue([0 <= x <= 1 for x in af.sigmoid(z)]) # check sigmoid implementation

        outputs = nn.feed_forward(sig_x[training_data_index])['a']
        self.assertEqual(len(outputs.shape), 1) # check if returns a vector, not a matrix
        self.assertEqual(outputs.shape[0], nn.layer_sizes[-1]) # check shape of the vector against the layer

        cost = NeuralNet.cost_feed(outputs, sig_cosh[training_data_index])

    def test_backpropagation(self):
        x = np.linspace(-4, 4, 1000)
        cosh = np.cosh(x)
        # make inputs the list of numpy arrays
        sig_x = [np.array([af.sigmoid(n), af.sigmoid(n**2)]) for n in x]
        sig_cosh = [np.array([af.sigmoid(n), af.sigmoid(n**2)]) for n in cosh]
        layer_sizes=[len(sig_x[0]), 3, 4, len(sig_cosh[0])]
        nn = NeuralNet(layer_sizes=layer_sizes, a_functions=[af.sigmoid for n in layer_sizes[1:]], a_functions_prime=[af.sigmoid_prime for n in layer_sizes[1:]]) # first layer created to match the input size
        training_data_index = np.random.randint(0, len(sig_x)) # random input from the data set
        d_grad_dict = nn._get_cost_gradient(sig_x[training_data_index], sig_cosh[training_data_index])

        for gw, w in zip(d_grad_dict['w grad'], nn.weights):
            self.assertEqual(gw.shape, w.shape)
        for gb, b in zip(d_grad_dict['b grad'], nn.biases):
            self.assertEqual(gb.shape, b.shape)
    
    def test_network_wb_update(self):
        inputs = [np.random.randn(4) for i in range(100)]
        desired_outputs = [np.random.randn(4) for i in range(100)]
        sig_inputs = [af.sigmoid(i) for i in inputs]
        sig_desired_outputs = [af.sigmoid(o) for o in desired_outputs]
        learning_rate = .05
        batch = [(i, o) for i, o in zip(sig_inputs, sig_desired_outputs)]

        layer_sizes = [len(inputs[0]), 3, 4, len(desired_outputs[0])]
        nn = NeuralNet(layer_sizes=layer_sizes, a_functions=[af.sigmoid for n in layer_sizes[1:]], a_functions_prime=[af.sigmoid_prime for n in layer_sizes[1:]]) # first layer created to match the input size

        old_shapes_w = [w.shape for w in nn.weights]
        old_shapes_b = [b.shape for b in nn.biases]
        mse = nn._update_weights_and_biases(batch, learning_rate)
        new_shapes_w = [w.shape for w in nn.weights]
        new_shapes_b = [b.shape for b in nn.biases]

        # check that update did not cause any dimension changes
        [self.assertEqual(w_old, w_new) for w_old, w_new in zip(old_shapes_w, new_shapes_w)]
        [self.assertEqual(b_old, b_new) for b_old, b_new in zip(old_shapes_b, new_shapes_b)]

    
    def test_learning(self):
        x = [np.array([np.random.uniform(-1, 1)]) for i in range(1000)]
        y = [np.cosh(n) for n in x]
        training_data = [(x, y) for x, y in zip(x, y)]

        layer_sizes = [1, 3, 1]
        nn = NeuralNet(layer_sizes=layer_sizes, a_functions=[af.sigmoid, af.linear], a_functions_prime=[af.sigmoid_prime, af.linear_prime]) # first layer created to match the input size
        print() # remove interference with other test prints (dots)
        nn.learn(labeled_training_dataset=training_data, no_epochs=10, mini_batch_size=10, learning_rate=.05) # can yield intermediate values


    def test_deserialization(self):
        pass
        # NeuralNet.load('test/resources/testnn.save') # deprecated, TODO need update

    def test_eval(sefl):
        x = [np.array([np.random.uniform(-1, 1)]) for i in range(1000)]
        y = [np.cosh(n) for n in x]

        training_data, test_data = NeuralNet.ratio_list_split([(x, y) for x, y in zip(x, y)], 0.75)
        
        neurons = [10, 3]
        epochs = [10, 3]
        learning_rates = [0.1, 0.05]

        params = itertools.product(neurons, epochs, learning_rates)
        print() # format so the print does not interfere with other test prints
        for p in params:
            mse_list = []
            nn = NeuralNet([1, p[0], 1], a_functions=[af.sigmoid, af.linear], a_functions_prime=[af.sigmoid_prime, af.linear_prime])
            print(p)
            for e in nn.gradient_descent_testdata(labeled_training_dataset=training_data, labeled_test_dataset=test_data, no_epochs=p[1], mini_batch_size=1, learning_rate=p[2]):
                print('EPOCH: {}\ttest mse: {}'.format(e['epoch'], e['test mse']), end='\r')
                mse_list.append(e['test mse'])
            print()



if __name__ == '__main__':
    unittest.main()