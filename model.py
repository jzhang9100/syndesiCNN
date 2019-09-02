from layers.conv2D import Conv2D
from layers.activation import Activation
from layers.dense import Dense
from layers.pool import MaxPool

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
        layer = "pool"
        self.layers.append(layer)

    #adds a fully connected layer with specified nuerons to layer queue
    def add_Dense(self, neurons):
        layer = "dense," + str(neurons)
        self.fully_connected_layers += 1
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
                score = []
                if batch == batches:
                    curr_batch = self.data[self.dat_size - (batch*batchSize):self.dat_size] 
                else:
                    curr_batch = self.data[batchSize*(batch-1):batch*batchSize] #deffintly a bertter way to do this lol
                
                print(curr_batch.shape)
                for tensor in curr_batch:
                    score.append(self.evaluate(tensor))
                    break
                break
    
    def evaluate(self, tensor):
        print(tensor, tensor.shape)
        def switch(layer, tensor):
            layers = {
                "conv":Conv2D(tensor, int(layer[1]), (int(layer[2]), int(layer[3]))).conv,
                "activation":Activation(tensor).activated_tensor,
                "pool":MaxPool(tensor).dwn_smpl,
                'dense':Dense(int(layer[1]), self.fully_connected_layers).out
                    }
            return layers[layer[0]]()
        
        results = tensor
        for layer in self.layers:
            l = layer.split(",")
            results = switch(l, results)
            print(results)
