import numpy as np

class Loss:
    '''
    y = binary labeld array
    p = predicted probability 
    '''
    def __init__(self, y, p)
        assert len(y) = len(p)
        self.label = y
        self.predicted = p
        self.cross_entropy = CrossEntropy()

    def CrossEntropy(self):
        loss = 0
        for i in range(len(self.label)):
            loss += self.label[i]*np.log(self.predicted[i])

        return -loss
