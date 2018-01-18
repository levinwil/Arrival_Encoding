import sys
sys.path.append("../functions")
import word_vector_map as wvm
import encoder
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

def generate_sentences():
    sentences = []
    vocab = wvm.get_vocab()
    for first_word in vocab:
        for second_word in vocab:
            sentences.append(first_word + " " + second_word)
    return sentences

def generate_sentence_image_map():
    sentences = generate_sentences()
    image_sentence_map = dict()
    for sentence in sentences:
        image_sentence_map[sentence] = encoder.encode(sentence)
    return image_sentence_map

if __name__ == "__main__":
    image_sentence_map = generate_sentence_image_map()
    sentences = image_sentence_map.keys()
    images = [image_sentence_map.get(sentence) for sentence in sentences]
    images_train, images_test, sentences_train, sentences_test = train_test_split(images, sentences, test_size=0.33)
    images_train = np.reshape(images_train, (len(images_train), 128, 128, 1))
    images_test = np.reshape(images_test, (len(images_test), 128, 128, 1))
    pickle.dump(images_train, open( "train.images", "wb" ) )
    pickle.dump(images_test, open( "test.images", "wb" ) )
    pickle.dump(sentences_train, open( "train.sentences", "wb" ) )
    pickle.dump(sentences_test, open( "test.sentences", "wb" ) )
