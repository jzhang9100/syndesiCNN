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
        #we apply the softmax function to the input to return the class probablities
        soft = self.softmax(self.flat)
        return soft

    def softmax(self, x):
        exp = np.exp(x)
        norm = np.sum(exp)
        exp = exp/norm
        exp = -1*np.log10(exp)
        print(exp, 'scores')
        return exp

    def fowardpass(self):
        assert len(self.bias) == len(self.weights)
        return np.maximum(0, np.dot(self.weights, self.flat) + self.bias)

class dense_test(unittest.TestCase):
    def test_softmax(self):
        scores = [3.2, 5.1, -1.7]
        d = Dense(3, scores)
        out = d.predict()
        assert round(out[0], 2) == 0.89

if __name__ == '__main__':
    unittest.main()
