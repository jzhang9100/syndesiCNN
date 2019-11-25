from layers.conv2D import Conv2D
from layers.activation import Activation
from layers.dense import Dense
from layers.pool import MaxPool
from backprop.loss import Loss

import numpy as np

class Sequential:
    def __init__(self, train_data, train_labels):
        assert len(train_data) == len(train_labels)
        self.data = train_data
        self.dat_size = len(self.data)
        self.labels = train_labels
        self.layers = []
        self.fully_connected_layers = 0
    
    #adds conv layer with parameters into layer queue
    def add_Conv(self, filters, kernel_size):
        x,y = kernel_size
        layer = "conv," + str(filters) + "," + str(x) + "," + str(y)
        self.layers.append(layer)
    
    #adds an activation layer to layer queue
    def add_Activation(self):
        layer = "activation"
        self.layers.append(layer)
    
    #adds a max pooling layer to the layer queue
    def add_Pool(self):
        layer = "pool,"
        self.layers.append(layer)

    #adds a fully connected layer with specified nuerons to layer queue
    def add_Dense(self, neurons):
        layer = "dense," + str(neurons)
        self.layers.append(layer)
    
    def out(self, neurons):
        layer = "output, " + str(neurons)
        self.layers.append(layer)
    
    #prints layer queue
    def show(self):
        for layer in self.layers:
            print(layer)
    
    '''
    trains the model with user given epochs and batch sizes
    can have a vaildation split if specified, other wise default is none
    '''
    def compile(self, epochs, batchSize, validation_split = None):
        for epoch in range(epochs):
            batches = self.dat_size//batchSize
            for batch in range(1,batches+1):
                curr_batch = None
                curr_scores = None 
                if batch == batches:
                    curr_batch = self.data[self.dat_size - (batch*batchSize):self.dat_size] 
                    curr_scores = self.labels[self.dat_size - (batch*batchSize):self.dat_size]  
                else:
                    curr_batch = self.data[batchSize*(batch-1):batch*batchSize] #deffintly a bertter way to do this lol
                    curr_scores = self.labels[batchSize*(batch-1):batch*batchSize]

                batch_pred = []
                for tensor in curr_batch:
                    batch_pred.append(self.evaluate(tensor))
                
                loss = Loss(batch_pred, curr_scores)
                print('loss', loss.get_cross_loss())
    
    def evaluate(self, tensor):
        def switch(layer, t):
            layer_type = layer[0]
            
            #make shift switch statement 
            if layer_type == "conv":
                c = Conv2D(t, int(layer[1]), (int(layer[2]), int(layer[3])))
                return c.conv
            
            elif layer_type == "activation":
                a = Activation(t)
                return a.activated_tensor
            
            elif layer_type == "pool":
                p = MaxPool(t)
                return p.pooled

            elif layer_type == "dense":
                d = Dense(int(layer[1]), t)
                return d.fpass
            
            elif layer_type == "output":
                d = Dense(int(layer[1]), t)
                return d.predict()
            else:
                print("fail")

        results = tensor
        for layer in self.layers:
            l = layer.split(",")
            results = switch(l, results)
        return results
