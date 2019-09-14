import numpy as np

#fully connected layer, takes activations and implements a neural network to compute class scores
class Dense:
    def __init__(self, neurons, tensor, num_layers):
        self.neurons = neurons  #number of neurons in the layer
        self.flat = np.array(tensor).flatten()  #flatten tensor
        #weights initiated to be random matrix of neurons x flattened input
        self.weights = np.random.randn(self.neurons,len(self.flat)) / np.sqrt(len(self.flat))
        self.bias = np.random.randn(self.neurons) * np.sqrt(2.0/self.neurons) #random bias vecotor with mean 0 and variance 1
        self.fpass = self.fowardpass()  #pass input, wieghts, and bias through sigmoid function
        self.out = self.predict()

    def predict(self):
        return np.dot(self.weights, self.flat) + self.bias

    def fowardpass(self):
        assert len(self.bias) == len(self.weights)
        return np.maximum(0, np.dot(self.weights, self.flat) + self.bias)

    def show(self):
        print('score', self.out, np.shape(self.out))
