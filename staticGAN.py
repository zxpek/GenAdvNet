# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
#%%
import numpy
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Reshape, BatchNormalization, LeakyReLU
from keras.optimizers import SGD, Adam
import keras
from matplotlib import pyplot as plt
#%%
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(64, activation='relu', input_shape=(784,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr = 0.01, momentum = 0.9)
model.compile(optimizer = sgd,
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])
#%%
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#%%


plt.imshow(x_train[0]/255)
plt.show()

#%%
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

x_train = x_train/255
x_test = x_test/255
#%%
model.fit(x_train, y_train, epochs = 30, batch_size = 30)

#%%
generator = Sequential()
generator.add(Dense(256, input_shape=(10,)))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(512))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(1024))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(784, activation='tanh'))
generator.add(Reshape((28,28)))


optimizer = Adam(0.0002, 0.5)

generator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])
        
        
#generator = Sequential()
#generator.add(Dense(64, activation='relu', input_shape=(10,)))
#generator.add(Dense(64, activation='relu'))
#generator.add(Dense(64, activation='relu'))
#generator.add(Dense(784, activation='relu'))
#generator.add(Reshape((28,28)))
#generator.compile(optimizer = sgd,
#              loss = 'binary_crossentropy',
#              metrics=['accuracy'])
#%%

generator.fit(y_train, x_train, epochs = 10, batch_size = 30)

#%%
#digit = 3
no1 = np.array([[0,0,0,0,0,0,0,1,0,0]])
#no1[0][digit] = 1
img = generator.predict(no1)
plt.imshow(img[0])
plt.show()
