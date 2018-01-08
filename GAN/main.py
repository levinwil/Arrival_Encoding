from keras import backend as K
K.set_image_dim_ordering('tf') # ensure our dimension notation matches
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import os
from PIL import ImageOps

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

generates NUM_PICTURES_TO_GENERATE images using the generator model and
saves those images to the ../alien_images directory

Parameters
____________
NUM_PICTURES_TO_GENERATE : int
    the number of pictures you'd like to generate

Return
____________
void
'''
def generate(NUM_PICTURES_TO_GENERATE = 1):
    #load the generator model
    g = __generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")

    #load the weights
    g.load_weights('../weights/generator_weights')

    #provide random noise
    noise = np.random.uniform(-1, 1, (NUM_PICTURES_TO_GENERATE, 100))

    #produce an array of alien image based off of the random noise
    generated_images = g.predict(noise, verbose=1)

    #access the images in that array, make them values such that they can be
    #visible, then save them
    for i in range(NUM_PICTURES_TO_GENERATE):
        generated_image = generated_images[i, :, :, 0]*127.5 + 127.5
        Image.fromarray(generated_image.astype(np.uint8)).save(
            "../alien_images/alien_image_" + str(i) + ".png")


if __name__ == "__main__":
    #parse the arguments
    parser = argparse.ArgumentParser(description='A general image generator.')
    parser.add_argument("--num_images", help="The number of images you'd like \
    to generate.", type=int)
    args = parser.parse_args()

    #if they didn't provide a number of images to generate, generate 1
    if args.num_images == None:
        generate(NUM_PICTURES_TO_GENERATE = 1)
    # if they did provide a number of images to generate, generate that many
    else:
        generate(NUM_PICTURES_TO_GENERATE = args.num_images)
