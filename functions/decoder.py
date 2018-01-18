import CNN
import word_vector_map
import numpy as np

'''
decode

Parameters
____________
image : numpy array
    the image you are decoding
num_words : Int
    the number of words encoded in the image

Return
____________
sentence : String
    the sentence encoded in the image
'''
def decode(image, num_words = 2):
    v = CNN.predict(image)[0:num_words][0]
    return word_vector_map.vec_to_sentence(v)
