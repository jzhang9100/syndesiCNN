from layers.conv2D import Conv2D
from layers.activation import Relu
from layers.dense import Dense
from layers.pool import MaxPool

class Sequential:
    def __init__(self, train_data, train_labels):
        assert len(train_data) == len(train_labels)
        self.data = train_data
        self.labels = train_labels
        self.layers = []

    def add_Conv(self, filters, kernel_size):
        layer = "conv," + str(filters) + "," + str(kernel_size)
        print(layer)
        self.layers.append(layer)

    def add_Activation(self):
        layer = "activation"
        self.layers.append(layer)

    def add_Pool(self):
        layer = "pool"
        self.layers.append(layer)

    def add_Dense(self, neurons):
        layer = "dense," + str(neurons)
        self.layers.append(layer)
