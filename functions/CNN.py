from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt

def __model(num_words = 2):
    input_shape = (128, 128, 1)
    model = Sequential([
        Conv2D(8, (3, 3), input_shape=input_shape, padding='same',
               activation='relu'),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        Conv2D(16, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        Conv2D(32, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        Conv2D(64, (3, 3), activation='relu', padding='same',),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(63, activation='relu'),
        Dense(num_words, activation='softmax')
    ])
    model.compile(optimizer='sgd', loss='categorical_crossentropy')
    return model

def train(x, y, epochs = 50, batch_size = 32, num_words = 2):
    model = __model(num_words = num_words)

    model_info = model.fit(x,
                           y,
                           epochs=epochs,
                           batch_size=batch_size)

    # save the weights
    model.save_weights("../weights/CNN_weights")

def predict(image):
    model = __model()
    model.load_weights("../weights/CNN_weights")
    return model.predict(np.reshape(image, (1, 128, 128, 1)))
