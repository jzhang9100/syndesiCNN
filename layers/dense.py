import numpy as np

def sigmoid(z): 
        #put z into sigmoid function
        sigmoid_function = 1 / (1 + np.exp(-z))
        return sigmoid_function

#fully connected layer, takes activations and implements a neural network to compute class scores
class Dense:
    def __init__(self, neurons, tensor):
        self.neurons = neurons  #number of neurons in the layer
        self.flat = np.array(tensor).flatten()  #flatten tensor
        #weights initiated to be random matrix of neurons x flattened input
        self.weights =  np.random.randint(-1, 2, (self.neurons, len(self.flat)))   
        self.bias = np.random.randint(-1, 2, (self.neurons, 1)) #random bias vecotor with mean 0 and variance 1
        self.out = self.fowardpass()  #pass input, wieghts, and bias through sigmoid function

        print('connected', self.out, np.shape(self.out))
        

    def fowardpass(self):
        out  = []

        assert len(self.bias) == len(self.weights) #make sure matrix multiplication finna work
        for i in range(self.neurons):
            z = np.dot(self.flat, self.weights[i]) + self.bias[i]  #sigmoid gate
            out.append(sigmoid(z[0]))
        return out
