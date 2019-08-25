from keras.datasets import cifar10
from layers.conv2D import Conv2D
from layers.activation import Activation

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)

x_train, x_test = x_train/255, x_test/255
test = x_train[0]
print(test)
conv_test = Conv2D(test, 64, (3,3))
#conv_test.get_kernel_info()

activation_test = Activation(conv_test.conv)
