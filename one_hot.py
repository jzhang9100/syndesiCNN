import numpy as np

def one_hot(y_data):
   y = np.array(y_data)
   y = y.ravel() # changes from [[1], [2], ..., [n]] to  [1, 2, 3,..,n]
   onehot = []
   num_classes = max(y) + 1
   
   for i in range(len(y)):
        categorical = [0] * num_classes # array of 0s with size of num_classes
        categorical[y[i]] = 1
        onehot.append(categorical)

   return onehot
