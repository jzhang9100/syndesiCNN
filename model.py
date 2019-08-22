from dense import Dense
from conv2D import Conv2D
from max_pooling import Maxpool
from normalization import batch_norm


class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer_type):

