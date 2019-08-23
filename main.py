from keras.datasets import cifar10
from layers.conv2D import Conv2D

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)

test = x_train[0]
conv_test = Conv2D(test, 64, (2,2))
