import numpy as np

'''
Takes in tensor 4-D  tensor with dimensions Length_Of_Data x Rows x Cols x Channels and performs a convolution resulting in
a new 4-D tensor with dimensions Length_Of_Data x newRows x newCols x Filters
'''

class Conv2D:
    def __init__(self, tensor, filters, kernel_size): 
        self.filters = filters
        self.tensor = tensor
        #get input size of image
        print(tensor.shape, tensor)
        print(self.tensor.shape, self.tensor)
        self.x, self.y, self.z = self.tensor.shape
        #establish kernel dimensions and assert that the kernel is a square
        (self.kernel_X, self.kernel_Y)  = kernel_size
        assert self.kernel_X == self.kernel_Y
        
        self.kernel_Z = self.z
        assert self.kernel_Z == len(self.tensor[0][0]) 

        self.kernel = [[np.randn(0, 255)]*self.kernel_X]
        print(self.kernel)
