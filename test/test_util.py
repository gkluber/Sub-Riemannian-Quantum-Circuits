import unittest
from unittest import TestCase
from subriemannian_qc.util import *
import numpy.testing as nptest
import itertools


class TestCombinations(TestCase):

    def test_zero(self):
        self.assertEqual([], combinations([], 10))
        self.assertEqual([], combinations([1, 2, 3, 4], 0))

    # Test combinations of only a single element
    def test_trivial(self):
        for x in range(1, 30):
            expected_array = np.array([x for _ in range(x)]).reshape(1, x)
            actual_array = combinations([x], x)
            nptest.assert_equal(actual_array, expected_array)

    def test_length(self):
        for basis in [range(x) for x in range(5)]:
            for length in range(5):
                expected_length = len(basis) ** length if length != 0 else 0
                actual_length = len(combinations(basis, length))
                self.assertEqual(actual_length, expected_length)

    '''
    def test_general(self):
        for basis in [range(x) for x in range(1, 5)]:
            for length in range(1, 5):
                product = list(itertools.product(basis, repeat=length))
                #product = [x[::-1] for x in product]
                expected_array = np.array(product)
                actual_array = combinations(basis, length)
                nptest.assert_equal(actual_array, expected_array)
    '''
