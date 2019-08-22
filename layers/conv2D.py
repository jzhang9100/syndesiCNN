
'''
Takes in tensor 4-D  tensor with dimensions Length_Of_Data x Rows x Cols x Channels and performs a convolution resulting in
a new 4-D tensor with dimensions Length_Of_Data x newRows x newCols x Filters
'''

class Conv2D:
    def __init__(self, tensor, filters, kernel_size, stride):
        self.filters = filters
        (self.kernel_X, self.kernel_Y)  = kernel_size
        self.stride = stride
        self.tensor = tensor
        
