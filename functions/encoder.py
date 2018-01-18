import GAN
import word_vector_map
import numpy as np
from PIL import Image

def pad_array_with_zeros(v):
    padded_v = v * (100 / len(v)) + [0] * (100 % len(v))
    return np.reshape(padded_v, (1, 100))

'''
encode

Parameters
____________
sentence : String
    the sentence you are encoding
save : Boolean
    if you want to save the image

Return
____________
image : python array
    the sentence encoded in an image
'''
def encode(sentence, save = True):
    v = word_vector_map.sentence_to_vec(sentence)
    padded_v = pad_array_with_zeros(v)
    generated_image = GAN.generate(padded_v)
    Image.fromarray(generated_image.astype(np.uint8)).save(
            "../alien_images/" + sentence.replace(" ", "_") + ".png")
    return generated_image
