#!/usr/bin/env python3

import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from PIL import Image
import os
from tensorflow.examples.tutorials.mnist import input_data as mnist
import shutil

class GAN(object):

    def __init__(self):
        self.discrim = None
        self.adv = None
        self.basic_gen = None
        self.basic_discrim = None
        self.z_dim = 100

    def basic_discriminator(self):
        if self.basic_discrim:
            return self.basic_discrim

        self.basic_discrim = Sequential()
        self.basic_discrim.add(Conv2D(32, (5,5), strides=(1,1), padding="SAME", input_shape=(28,28,1)))
        self.basic_discrim.add(Activation("relu"))
        self.basic_discrim.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="SAME"))
        self.basic_discrim.add(Dropout(0.3))

        self.basic_discrim.add(Conv2D(64, (5,5), strides=(1,1), padding="SAME"))
        self.basic_discrim.add(Activation("relu"))
        self.basic_discrim.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="SAME"))
        self.basic_discrim.add(Dropout(0.3))

        self.basic_discrim.add(Flatten())
        self.basic_discrim.add(Dense(1024))
        self.basic_discrim.add(Dense(1))
        self.basic_discrim.add(Activation("sigmoid"))
        
        return self.basic_discrim

    def basic_generator(self):
        if self.basic_gen:
            return self.basic_gen

        self.basic_gen = Sequential()
        self.basic_gen.add(Dense(9800, input_dim=100))
        self.basic_gen.add(BatchNormalization())
        self.basic_gen.add(Activation("relu"))
        self.basic_gen.add(Reshape((14,14,50)))

        self.basic_gen.add(UpSampling2D())
        self.basic_gen.add(Conv2DTranspose(100, (5,5), strides=(1,1), padding="SAME"))
        self.basic_gen.add(BatchNormalization())
        self.basic_gen.add(Activation("relu"))

        self.basic_gen.add(Conv2DTranspose(150, (5,5), strides=(1,1), padding="SAME"))
        self.basic_gen.add(BatchNormalization())
        self.basic_gen.add(Activation("relu"))

        self.basic_gen.add(Conv2DTranspose(1, (5,5), strides=(1,1), padding="SAME"))
        self.basic_gen.add(Activation("sigmoid"))
    
        return self.basic_gen

    def discriminator(self):
        if self.discrim:
            return self.discrim
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.discrim = Sequential()
        self.discrim.add(self.basic_discriminator())
        self.discrim.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        
        return self.discrim

    def adversarial(self):
        if self.adv:
            return self.adv
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.adv = Sequential()
        self.adv.add(self.basic_generator())
        self.adv.add(self.basic_discriminator())
        self.adv.layers[1].trainable = False
        self.adv.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return self.adv
            

class GAN_Trainer(object):
    def __init__(self):
        self.gan = GAN()
        self.discrim = self.gan.discriminator()
        self.gen = self.gan.basic_generator()
        self.adv = self.gan.adversarial()
        self.z_dim = 100

    def train(self, epochs, batch_size):
        x_train = mnist.read_data_sets("mnist", one_hot=True).train.images
        x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
        for i in range(epochs):
            indices = np.random.randint(0, x_train.shape[0], batch_size)
            d_noise = np.random.uniform(-1.0, 1.0, size=(batch_size, self.z_dim))
            real_imgs = x_train[indices,:,:,:]
            fake_imgs = self.gen.predict(d_noise)
            d_input = np.concatenate((fake_imgs, real_imgs))
            d_output = np.concatenate((np.zeros((batch_size,1)), np.ones((batch_size,1))))
            d_loss = self.discrim.train_on_batch(d_input, d_output)
            
            a_input = np.random.uniform(-1.0, 1.0, size=(batch_size, self.z_dim))
            a_output = np.ones((batch_size, 1))
            a_loss = self.adv.train_on_batch(a_input, a_output)

            if i % 50 == 0:
                print("d_loss: " + str(d_loss[0]) + "   " + "a_loss: " + str(a_loss[0]))
            if i % 100 == 0:
                test_noise = np.random.uniform(-1.0, 1.0, size=(1, self.z_dim))
                gen_img = self.gen.predict(test_noise)
                gen_img = np.reshape(gen_img, (28, 28)) * 255
                gen_img = np.array(gen_img, dtype='uint8')
                img_filename = "IMG/" + "batch" + str(i) + ".png"
                img = Image.fromarray(gen_img)
                img.save(img_filename)
            if i % 500 == 0:
                model_filename = "model/batch_" + str(i+1) + ".h5"
                self.gen.save(model_filename)


if(os.path.exists(os.getcwd() + "/IMG")):
    print("path exists")
    shutil.rmtree(os.getcwd() + "/IMG")
if(os.path.exists(os.getcwd() + "/model")):
    shutil.rmtree(os.getcwd() + "/model")
os.mkdir(os.getcwd() + "/IMG")
os.mkdir(os.getcwd() + "/model")
trainer = GAN_Trainer()
trainer.train(50000, 256)
