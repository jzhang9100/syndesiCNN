from layers.conv2D import Conv2D


class Sequential:
    def __init__(self, train_data, train_labels):
        assert len(train_data) == len(train_labels)
        self.data = train_data
        self.labels = train_labels
        self.layers = []

    def addConv(self, filters, kernel_size):
        layer = "conv," + str(filters) + "," + str(kernel_size)
        print(layer)
        self.layers.append(layer)
