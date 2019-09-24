#!/usr/bin/python3
import numpy as np
import unittest

#fully connected layer, takes activations and implements a neural network to compute class scores
class Dense:
    def __init__(self, neurons, tensor):
        self.neurons = neurons  #number of neurons in the layer
        self.flat = np.array(tensor).flatten()  #flatten tensor
        #weights initiated to be random matrix of neurons x flattened input
        self.weights = np.random.randn(self.neurons,len(self.flat)) / np.sqrt(len(self.flat))
        self.bias = np.random.randn(self.neurons) * np.sqrt(2.0/self.neurons) #random bias vecotor with mean 0 and variance 1
        self.fpass = self.fowardpass()  #pass input, wieghts, and bias through sigmoid function

    def predict(self):
        out = np.maximum(0, np.dot(self.weights, self.flat) + self.bias)
        print(out)
        soft = self.softmax(out)
        print(soft)
        return soft

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def fowardpass(self):
        assert len(self.bias) == len(self.weights)
        return np.maximum(0, np.dot(self.weights, self.flat) + self.bias)

class dense_test(unittest.TestCase):
    
    def test_softmax(self):
        self.assertEqual(1, 1)

    if __name__ == '__main__':
        unittest.main()
