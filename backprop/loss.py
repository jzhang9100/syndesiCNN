from math import log

class Loss:
    '''
    y = onehot label'd scores
    p = predicted probability 
    '''
    def __init__(self, y, p):
        assert len(y) == len(p)
        self.y = y
        self.p = p
        self.cross_entropy = self.CrossEntropy(self.y, self.p)

    def CrossEntropy(self, y, p):
        sum_score = 0.0 
        assert len(y) == len(p), "input arrays error, lengths dont match"
        
        for i in range(len(y)):
            for j in range(len(y[0])):
                sum_score += y[i][j] * (log(1e-15 + p[i][j]))
        mean_sum_score = (1.0 / len(y)) * sum_score
        return -mean_sum_score
