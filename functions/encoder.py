import sys
import GAN
import word_vector_map
import numpy as np

def __pad_array_with_zeros(v):
    num_zeros_to_add = 100 - len(v)
    padded_v = v + [0] * num_zeros_to_add
    return np.reshape(padded_v, (1, 100))

def encode(sentence):
    v = word_vector_map.sentence_to_vec(sentence)
    padded_v = __pad_array_with_zeros(v)
    GAN.generate(padded_v)

if __name__ == "__main__":
    encode("this is another sentence")
