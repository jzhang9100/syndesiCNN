import numpy as np

x = [100, 40, 30, 90]
y = [3, 1,2, 5]
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(x))
print(softmax(y))
