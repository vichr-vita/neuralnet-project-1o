from typing import Type
import numpy as np
import unittest

from numpy.core.arrayprint import _leading_trailing
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
        d_grad_dict = nn.get_cost_gradient(sig_x[training_data_index], sig_cosh[training_data_index])

        for gw, w in zip(d_grad_dict['w grad'], nn.weights):
            self.assertEqual(gw.shape, w.shape)
        for gb, b in zip(d_grad_dict['b grad'], nn.biases):
            self.assertEqual(gb.shape, b.shape)
    
    def test_network_wb_update(self):
        inputs = [np.random.randn(4) for i in range(100)]
        desired_outputs = [np.random.randn(4) for i in range(100)]
        sig_inputs = [NeuralNet.sigmoid(i) for i in inputs]
        sig_desired_outputs = [NeuralNet.sigmoid(o) for o in desired_outputs]
        learning_rate = .05
        batch = [(i, o) for i, o in zip(sig_inputs, sig_desired_outputs)]

        layer_sizes = [len(inputs[0]), 3, 4, len(desired_outputs[0])]
        nn = NeuralNet(layer_sizes=layer_sizes)

        old_shapes_w = [w.shape for w in nn.weights]
        old_shapes_b = [b.shape for b in nn.biases]
        mse = nn.update_weights_and_biases(batch, learning_rate)
        new_shapes_w = [w.shape for w in nn.weights]
        new_shapes_b = [b.shape for b in nn.biases]

        # check that update did not cause any dimension changes
        [self.assertEqual(w_old, w_new) for w_old, w_new in zip(old_shapes_w, new_shapes_w)]
        [self.assertEqual(b_old, b_new) for b_old, b_new in zip(old_shapes_b, new_shapes_b)]

    
    def test_learning(self):
        inputs = [np.random.randn(4) for i in range(10000)]
        desired_outputs = [np.random.randn(4) for i in range(10000)]
        sig_inputs = [NeuralNet.sigmoid(i) for i in inputs]
        sig_desired_outputs = [NeuralNet.sigmoid(o) for o in desired_outputs]
        training_data = [(i, o) for i, o in zip(sig_inputs, sig_desired_outputs)]

        layer_sizes = [len(inputs[0]), 3, len(desired_outputs[0])]
        nn = NeuralNet(layer_sizes=layer_sizes)
        nn.learn(labeled_training_dataset=training_data, no_epochs=1000, mini_batch_size=1000, learning_rate=.05)

if __name__ == '__main__':
    unittest.main()