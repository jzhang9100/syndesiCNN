import numpy as np

'''
Takes in tensor 4-D  tensor with dimensions Length_Of_Data x Rows x Cols x Channels as input. 
Adds padding to the tensor and generates the weight matrix. Performs convolution with stride of 1. 
Performs a convolution resulting in a new 4-D tensor with dimensions Length_Of_Data x newRows x newCols x Filters
'''

class Conv2D:
    def __init__(self, tensor, filters, kernel_size): 
        self.filters = filters
        self.tensor = tensor
        (self.x, self.y, self.z) = self.tensor.shape

        self.kernel = []
        (self.kernX, self.kernY) = kernel_size
        self.kernZ = self.z
        assert self.kernX == self.kernY  #make sure filter is square
        assert self.kernZ == len(self.tensor[0][0])  #make sure kernel and tensor have the same depth 
       
        #create 'filters' amount of filters
        for i in range(filters):
            k = self.get_filter()
            self.kernel.append(k)
        assert len(self.kernel) == filters  #make sure there is the right amount of filters

        #set tensor and apply 'same' padding to tensor
        self.padded_tensor = self.same_padding(tensor)     
        (self.padX, self.padY, self.padZ) = self.padded_tensor.shape
        print('padded tensor', self.padded_tensor.shape, self.padded_tensor)
        
        self.conv = self.conv2D()
    
    def conv2D(self):
        #first try to print each part of the tensor that needs to be multiplied
        for r in range(3, self.padX+1):
            for c in range(3, self.padY+1):
                sec = self.padded_tensor[r-3:r,c-3:c]
                print('conv',sec, sec.shape)
                break

    #generates random filter with dimensions kernX x kernY x kernZ, initiates values to be integers [-1, 2) aka [-1, 1]
    def get_filter(self):
        kern = np.random.randint(-1, 2, (self.kernZ, self.kernY, self.kernX)).tolist()        
        return kern

    #gets kernel size and mean val
    def get_kernel_info(self):
        print("Size", self.kernX, self.kernY, self.kernZ)
        
        def findKernelMean(kern):
            sum = 0
            for x in range(self.kernZ):
                for y in range(self.kernX):
                    for z in range(self.kernY):
                        sum += kern[x][y][z]
            return sum/(self.kernX*self.kernY*self.kernZ)
        
        kernelInfo = [findKernelMean(x) for x in self.kernel]
        #peek kernel
        print(self.kernel[0])
        #we want to make sure we have around a zero mean
        print("Mean Vals: ", kernelInfo)
    
    #applys 'same' padding to tensor
    def same_padding(self, tensor):
        padX, padY = self.kernX - 2, self.kernY - 2
        t = np.pad(tensor, (padX, padY), 'constant')
        return t[:,:,1:4] 
