#!/usr/bin/python3
from keras.datasets import cifar10
from model import Sequential
from layers.pool import MaxPool

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train/255, x_test/255
test = x_train[0]

model = Sequential(x_train, y_train)

model.add_Conv(32, (3,3))
model.add_Activation()
model.add_Pool()

model.add_Conv(32, (3,3))
model.add_Activation()
model.add_Pool()

model.add_Conv(64, (3,3))
model.add_Activation()
model.add_Pool()

model.add_Dense(512)
model.add_Dense(10)

model.compile(1, 32) 

