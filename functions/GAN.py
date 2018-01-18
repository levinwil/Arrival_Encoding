from keras import backend as K
K.set_image_dim_ordering('tf') # ensure our dimension notation matches
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
import numpy as np
import argparse
import os
import sys
sys.path.append("../utils")
import word_vector_map as wvm

'''
generator_model

Parameters
____________
None

Return
____________
model : keras model
    The generator model
'''
def __generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*8*8))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((8, 8, 128), input_shape=(128*8*8,)))
    model.add(UpSampling2D(size=(8, 8)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model

'''
generate

generates one image using the generator model based off of the input vector

Parameters
____________
v : Numpy array (must be 1 x 100)
    the vector being used to generate the image

Return
____________
generated_image : numpy array
'''
def generate(v):
    #load the generator model
    g = __generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")

    #load the weights
    g.load_weights('../weights/generator_weights')

    #produce an array of alien image based off of the vector
    generated_images = g.predict(v, verbose=1)

    return generated_images[0, :, :, 0]*127.5 + 127.5
