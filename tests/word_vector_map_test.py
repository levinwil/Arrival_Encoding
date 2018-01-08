import unittest
import sys
sys.path.append("../functions/word_vector_map")
from main import *

class TestWordVectorMap(unittest.TestCase):

    def test_word_to_num_map(self):
        actual = {0.0: 'and', 0.5: 'one', 0.7142857142857143: 'the', 0.7857142857142857: 'more', 0.14285714285714285: 'final', 0.07142857142857142: 'word2vec', 0.5714285714285714: 'second', 0.21428571428571427: 'for', 0.42857142857142855: 'is', 0.6428571428571429: 'another', 0.9285714285714286: 'first', 0.2857142857142857: 'sentence', 0.8571428571428571: 'yet', 0.35714285714285715: 'this'}
        self.assertEqual(generate_num_to_word_map(), actual)

    def test_num_to_word_map(self):
        actual = {'and': 0.0, 'word2vec': 0.07142857142857142, 'for': 0.21428571428571427, 'sentence': 0.2857142857142857, 'this': 0.35714285714285715, 'is': 0.42857142857142855, 'one': 0.5, 'second': 0.5714285714285714, 'another': 0.6428571428571429, 'the': 0.7142857142857143, 'first': 0.9285714285714286, 'yet': 0.8571428571428571, 'final': 0.14285714285714285, 'more': 0.7857142857142857}
        self.assertEqual(generate_word_to_num_map(), actual)

    def test_sentence_to_vec(self):
        vector = [0.35714285714285715, 0.42857142857142855, 0.6428571428571429, 0.2857142857142857]
        sentence = "this is another sentence"
        self.assertEqual(sentence_to_vec(sentence), vector)

    def test_vec_to_sentence(self):
        vector = [0.35714285714285715, 0.42857142857142855, 0.6428571428571429, 0.2857142857142857]
        sentence = "this is another sentence"
        self.assertEqual(vec_to_sentence(vector), sentence)

    def test_two_way_conversion_vec(self):
        vector = [0.35714285714285715, 0.42857142857142855, 0.6428571428571429, 0.2857142857142857]
        self.assertEqual(sentence_to_vec(vec_to_sentence(vector)), vector)

    def test_two_way_conversion_sentence(self):
        sentence = "this is another sentence"
        self.assertEqual(vec_to_sentence(sentence_to_vec(sentence)), sentence)

if __name__ == '__main__':
    unittest.main()
