'''
Takes in tensor and applies an activation function (only using ReLU right now but more to add). Returns tensor with activation function applied
'''

class Activation:
    def __init__(self, tensor):
        self.tensor = tensor
        self.activated_tensor = self.relu()

    def relu(self):
        t = self.tensor
        max_fn = lambda x : max(0, x)  #ReLU function
        for x, d in enumerate(t):
            for i, l in enumerate(d):
                t[x][i] = list(map(max_fn, l))
        return t
