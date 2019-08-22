import numpy as np

class sigmoid:
    def __init__(self, x, w, b):
        assert len(x) == len(w) == len(b)
        self.x = x #input vector
        self.w = w #weights vector
        self.bias = b #biases for each sigmoid
        
        #calculate Z as input to sigmoid function
        sum = 0
        for i in range(len(x)):
            sum += self.x[i] * self.w[i] + self.bias[i]
        self.z = sum
        
        #put z into sigmoid function
        self.sigmoid_function = 1 / (1 + np.exp(-self.z))
    
    def get_output(self):
        return self.sigmoid_function

class Dense:
    def __init__(self, dimensionality):
        self.units = dimensionality
        
