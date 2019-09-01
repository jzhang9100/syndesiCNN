#!/usr/bin/python3
from keras.datasets import cifar10
from model import Sequential

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train/255, x_test/255
model = Sequential(x_train, y_train)
model.addConv(32, (3,3))
