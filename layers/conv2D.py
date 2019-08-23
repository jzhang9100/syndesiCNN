import numpy as np

'''
Takes in tensor 4-D  tensor with dimensions Length_Of_Data x Rows x Cols x Channels as input. Adds padding to the tensor and generates the weight matrix. Performs a convolution resulting in a new 4-D tensor with dimensions
Length_Of_Data x newRows x newCols x Filters
'''

class Conv2D:
    def __init__(self, tensor, filters, kernel_size): 
        self.filters = filters
        self.tensor = tensor
        #get input size of image
        print(self.tensor.shape, self.tensor)
        print(len(self.tensor), len(self.tensor[0]), len(self.tensor[0][0]))
        self.x, self.y, self.z = self.tensor.shape
        #establish kernel dimensions and assert that the kernel is a square
        (self.kernX, self.kernY)  = kernel_size
        assert self.kernX == self.kernY
        #make sure kernel depth is equal to image depth 
        self.kernZ = self.z
        assert self.kernZ == len(self.tensor[0][0]) 
        #make this a new function 
        #generates random filter with dimensions kernX x kernY x kernZ, initiates values to be integers [-1, 2) aka [-1, 1]
        self.kernel = np.random.randint(-1, 2, (self.kernZ, self.kernY, self.kernX)).tolist() 
                        
   #def same_padding(self, tensor):
    def get_kernel(self):
        print("Size", self.kernX, self.kernY, self.kernZ)
        print("Kernel:", self.kernel)
        def findKernelMean():
            sum = 0
            for x in range(self.kernZ):
                for y in range(self.kernX):
                    for z in range(self.kernY):
                        sum += self.kernel[x][y][z]
            return sum
        print("Mean Val: ", findKernelMean())
