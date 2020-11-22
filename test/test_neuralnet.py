from typing import Type
import numpy as np
import unittest
from neural_net.neuralnet import NeuralNet, NeuralNetException

class NeuralNetTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass


    def test_basic(self):
        x = np.linspace(-4, 4, 1000)
        cosh = np.cosh(x)
        # make inputs the list of numpy arrays
        sig_x = [np.array([NeuralNet.sigmoid(n)]) for n in x]
        sig_cosh = [np.array([NeuralNet.sigmoid(n)]) for n in cosh]

        nn = NeuralNet(layer_sizes=[len(sig_x[0]), 3, 4, 1], initialize=False) # first layer created to match the input size
        self.assertTrue(len(nn.biases) == len(nn.layer_sizes) - 1)
        for b, s in zip(nn.biases, nn.layer_sizes[1:]):
            self.assertEqual(len(b), s)
        nn.initialize_weights()
        nn.initialize_biases()
        # print(nn.weights)

        wrong_type = 3
        self.assertRaises(TypeError, NeuralNet.z_func, wrong_type, nn.weights[0], nn.biases[0])
        
        layer_no = 1 # second layer
        training_data_index = np.random.randint(0, len(sig_x)) # random input from the data set

        z = NeuralNet.z_func(sig_x[training_data_index], nn.weights[layer_no - 1], nn.biases[layer_no - 1])
        self.assertEqual(len(z.shape), 1) # check if returns a vector, not a matrix
        self.assertEqual(z.shape[0], nn.layer_sizes[layer_no]) # check shape of the vector agains the layer
        self.assertTrue([0 <= x <= 1 for x in NeuralNet.sigmoid(z)]) # check sigmoid implementation

        outputs = nn.feed_forward(sig_x[training_data_index])
        self.assertEqual(len(outputs.shape), 1) # check if returns a vector, not a matrix
        self.assertEqual(outputs.shape[0], nn.layer_sizes[-1]) # check shape of the vector against the layer

        cost = NeuralNet.cost_feed(outputs, sig_cosh[training_data_index])
        # print(cost)

    def test_backpropagation(self):
        x = np.linspace(-4, 4, 1000)
        cosh = np.cosh(x)
        # make inputs the list of numpy arrays
        sig_x = [np.array([NeuralNet.sigmoid(n), NeuralNet.sigmoid(n**2)]) for n in x]
        sig_cosh = [np.array([NeuralNet.sigmoid(n), NeuralNet.sigmoid(n**2)]) for n in cosh]
        nn = NeuralNet(layer_sizes=[len(sig_x[0]), 3, 4, len(sig_cosh[0])]) # first layer created to match the input size
        training_data_index = np.random.randint(0, len(sig_x)) # random input from the data set
        grad_weights, grad_biases = nn.get_cost_gradient(sig_x[training_data_index], sig_cosh[training_data_index])

        for gw, w in zip(grad_weights, nn.weights):
            self.assertEqual(gw.shape, w.shape)
        for gb, b in zip(grad_biases, nn.biases):
            self.assertEqual(gb.shape, b.shape)
    
    def test_network_wb_update(self):
        nn.update(inputs, desired_outputs, learning_rate)

if __name__ == '__main__':
    unittest.main()