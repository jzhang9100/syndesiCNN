import numpy as np
'''
Takes in tensor and performs max pooling to reduce spacial size and to reduce the computations needed in the neural network. Down 
samples using a spacial extent of 2x2 andd stride of 1. Ex. input volume of 64x32x32 should return output volume 64x16x16
'''

class MaxPool:
    def __init__(self, tensor):
        self.tensor = tensor
        (self.z, self.y, self.x) = self.tensor.shape
        self.pooled = self.maxpool()

    def maxpool(self):
        t = self.tensor
        down_sampled = []
        #loop through tensor and 
        for i, l in enumerate(t):
            max_filter = []
            r = 2
            while r <= self.y:
                max_region = [] 
                #for m, p x q matrcies in the tensor, find and compute the maxpool of the 2 x 2 region
                c = 2
                while c  <= self.x:
                    local_region = self.tensor[i,r-2:r,c-2:c]
                    m = max(local_region.flatten())
                    max_region.append(m)
                    c += 2

                max_filter.append(max_region)
                r +=2 
            
            down_sampled.append(max_filter)
        down_sampled = np.array(down_sampled)
        return down_sampled
