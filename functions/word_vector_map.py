#NOTE: might need these if raw mapping does not work
'''
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
'''

def __generate_word_num_tuples():
    vocab_set = list(set(['this', 'is', 'the', 'first', 'sentence', 'for',
                'word2vec', 'this', 'is', 'the', 'second', 'sentence',
    			'yet', 'another', 'sentence',
    			'one', 'more', 'sentence',
    			'and', 'the', 'final', 'sentence']))

    num_vocab = len(vocab_set)
    vocab_tuples = [(vocab_set[i], i * 1.0 / num_vocab) for i in range(num_vocab)]
    return vocab_tuples

'''
generate_word_to_num_map

Parameters
____________
None

Return
____________
dictionary : dictionary
    The dictionary map from words to numbers
'''
def generate_word_to_num_map():
    return dict(__generate_word_num_tuples())


'''
generate_num_to_word_map

Parameters
____________
None

Return
____________
dictionary : dictionary
    The dictionary map from numbers to words
'''
def generate_num_to_word_map():
    reverse_map = __generate_word_num_tuples()
    return dict((v, k) for k, v in reverse_map)


'''
sentence_to_vec

Parameters
____________
sentence : String
    the sentence you want to represent as a vector

Return
____________
v : vector
    the vector representation of the sentence
'''
def sentence_to_vec(sentence):
    word_array = sentence.split(' ')
    vocab_map = generate_word_to_num_map()
    return map(lambda x: vocab_map.get(x), word_array)


'''
vec_to_sentence

Parameters
____________
v : Numpy array
    the vector you want to convert to a sentence

Return
____________
s : sentence
    the sentence representation of the vector
'''
def vec_to_sentence(v):
    vocab_map = generate_num_to_word_map()
    word_array = map(lambda x: vocab_map.get(x), v)
    return " ".join(word_array)
