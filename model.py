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
    
    #adds conv layer with parameters into layer queue
    def add_Conv(self, filters, kernel_size):
        layer = "conv," + str(filters) + "," + str(kernel_size)
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
        for epoch in epochs:
            batches = self.dat_size//batchSize
            for batch in batches:
                curr_batch = None
                if batch == batches:
                    curr_batch = [self.dat_size:self.dat_size - (batch*batchSize)] 
                else:
                    curr_batch = self.data[batch*batchSize:batch*(batchSize-1)] #deffintly a bertter way to do this lol
                print(curr_batch.shape)
