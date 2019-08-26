'''
Takes in tensor and performs max pooling to reduce spacial size and to reduce the computations needed in the neural network. Down 
samples using a spacial extent of 2x2 andd stride of 1. Ex. input volume of 64x32x32 should return output volume 64x16x16
'''

class MaxPool:
    def __init__(self, tensor):
        self.tensor = tensor
        
        self.dwn_smpl = self.maxpool()

    def maxpool(self):
        t = self.tensor
        for i in range(len(t)):
            print(t[i])
