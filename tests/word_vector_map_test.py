import unittest
import sys
sys.path.append("../functions")
from word_vector_map import *

class TestWordVectorMap(unittest.TestCase):

    def test_get_vocab(self):
        actual = set(['cat', 'tree', 'apple', 'car', 'mouse'])
        self.assertEqual(set(get_vocab()), actual)

    def test_word_to_num_map(self):
        actual = {0.0: 'cat', 0.8: 'mouse', 0.6: 'car', 0.4: 'apple', 0.2: 'tree'}
        self.assertEqual(generate_num_to_word_map(), actual)

    def test_num_to_word_map(self):
        actual = {'cat': 0.0, 'mouse': 0.8, 'car':0.6, 'apple':0.4, 'tree': 0.2}
        self.assertEqual(generate_word_to_num_map(), actual)

    def test_sentence_to_vec(self):
        vector = [0.0, 0.2]
        sentence = "cat tree"
        self.assertEqual(sentence_to_vec(sentence), vector)

    def test_vec_to_sentence(self):
        vector = [0.0, 0.2]
        sentence = "cat tree"
        self.assertEqual(vec_to_sentence(vector), sentence)

    def test_two_way_conversion_vec(self):
        vector = [0.0, 0.2]
        self.assertEqual(sentence_to_vec(vec_to_sentence(vector)), vector)

    def test_two_way_conversion_sentence(self):
        sentence = "cat tree"
        self.assertEqual(vec_to_sentence(sentence_to_vec(sentence)), sentence)

if __name__ == '__main__':
    unittest.main()
