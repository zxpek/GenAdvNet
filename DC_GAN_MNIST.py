# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 02:58:32 2018

@author: zhpek
"""
import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Reshape, BatchNormalization, LeakyReLU, Input, UpSampling2D, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
import keras
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.layers.core import Activation

class gan():
    def __init__(self):
        self.image_shape = (28,28,1)
        self.noise_shape = (100,)
        self.optimizer = Adam(0.0002, 0.5)
        
        self.generator = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.stacked = self.stackGAN()

    def stackGAN(self):
        noise_in = Input(shape=self.noise_shape)
        gen_image = self.generator(noise_in)
        self.discriminator.trainable = False
        decision = self.discriminator(gen_image)
        stacked = Model(noise_in, decision)
        stacked.compile(loss = 'binary_crossentropy', optimizer = self.optimizer)
        return(stacked)
        
    def init_generator(self):
        
        generator = Sequential()
        generator.add(Dense(1024, input_shape=self.noise_shape))
        generator.add(Activation('tanh'))
        generator.add(Dense(128*7*7))
        generator.add(BatchNormalization())
        generator.add(Activation('tanh'))
        generator.add(Reshape((7, 7, 128)))
        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(64, (5, 5), padding='same'))
        generator.add(Activation('tanh'))
        generator.add(UpSampling2D(size=(2, 2)))
        generator.add(Conv2D(1, (2, 2), padding='same'))
        generator.add(Activation('tanh'))
        generator.add(Reshape(self.image_shape))
        
        generator.compile(loss = 'binary_crossentropy', 
                    optimizer = self.optimizer)
        
        return(generator)
        
    def init_discriminator(self):
        discriminator = Sequential()
        discriminator.add(Flatten(input_shape = self.image_shape))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(128))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(64))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(32))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(32))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(32))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(32))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(32))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(32))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(Dense(1, activation='sigmoid'))
        
        discriminator.compile(loss = 'binary_crossentropy', 
                    optimizer = self.optimizer,
                    metrics=['accuracy'])
        return(discriminator)
        
    def reshape(self, in_obj = None, no_shape = None):
        if in_obj:
            self.input_shape = in_obj.shape
            
        if no_shape:
            self.noise_shape = no_shape
            
    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()
        #plt.imshow(X_train[0])
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.stacked.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        
        
        fig.savefig(os.path.join(os.path.expanduser('~'),"mnist_%d.png" % epoch), dpi = 300)
        plt.close()


if __name__ == '__main__':
    g = gan()
    g.train(epochs=30000, batch_size=32, save_interval=50)