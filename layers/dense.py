import numpy as np

def relu(z):
    return max(0, z)

#fully connected layer, takes activations and implements a neural network to compute class scores
class Dense:
    def __init__(self, neurons, tensor, num_layers):
        self.neurons = neurons  #number of neurons in the layer
        self.flat = np.array(tensor).flatten()  #flatten tensor
        #weights initiated to be random matrix of neurons x flattened input
        self.weights =  np.random.randint(-1, 2, (self.neurons, len(self.flat)))   
        self.bias = np.random.normal(0, np.sqrt(2.0/num_layers), (self.neurons, 1)) #random bias vecotor with mean 0 and variance 1
        self.out = self.fowardpass()  #pass input, wieghts, and bias through sigmoid function

    def fowardpass(self):
        out  = []

        assert len(self.bias) == len(self.weights) #make sure matrix multiplication finna work
        for i in range(self.neurons):
            z = np.dot(self.flat, self.weights[i]) + self.bias[i]  
            out.append(relu(z))
        return out

    def show(self):
        print('score', self.out, np.shape(self.out))
