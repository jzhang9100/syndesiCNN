import numpy as np

class sigmoid:
    def __init__(self, z):
        self.z = z #input 
        
        #put z into sigmoid function
        self.sigmoid_function = 1 / (1 + np.exp(-self.z))
    
    def get_output(self):
        return self.sigmoid_function

#fully connected layer, takes activations and implements a neural network to compute class scores
class Dense:
    def __init__(self, neurons, tensor):
        self.neurons = neurons  #number of neurons in the layer
        self.flat = np.array(tensor).flatten()  #flatten tensor
        #weights initiated to be random matrix of neurons x flattened input
        self.weights =  np.random.randint(-1, 2, (self.neurons, len(self.flat)))   
        self.bias = np.random.randint(-1, 2, (self.neurons, 1)) #random bias vecotor with mean 0 and variance 1
        
        self.fowardpass()

    def fowardpass(self):
        print('weights', self.weights, self.weights.shape)
        print('bias', self.bias, self.bias.shape)
        
