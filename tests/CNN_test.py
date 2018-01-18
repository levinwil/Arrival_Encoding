import unittest
import sys
sys.path.append("../functions")
from decoder import decode
import pickle
import numpy.testing as np

class TestWordVectorMap(unittest.TestCase):


    def test_generator_test_data(self):
        X = pickle.load( open( "../data/test.images", "rb" ) )
        actual_y = pickle.load( open( "../data/test.sentences", "rb" ) )
        calculated_y = [decode(image) for image in X]
        np.assert_array_equal(calculated_y, actual_y)

    def test_generator_train_data(self):
        X = pickle.load( open( "../data/train.images", "rb" ) )
        actual_y = pickle.load( open( "../data/train.sentences", "rb" ) )
        calculated_y = [decode(image) for image in X]
        np.assert_array_equal(calculated_y, actual_y)


if __name__ == '__main__':
    unittest.main()
