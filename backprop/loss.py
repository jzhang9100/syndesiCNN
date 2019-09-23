import numpy as np

class Loss:
    '''
    y = binary labeld array
    p = predicted probability 
    '''
    def __init__(self, y, p):
        assert len(y) == len(p)
        self.cross_entropy = self.CrossEntropy(y, p)

    def CrossEntropy(self, y, p):
        loss = 0
        for i in range(len(y)):
            loss += np.nan_to_num(-y[i]*np.log(p[i])-(1-y[i])*np.log(1-p[i]))
        return loss
