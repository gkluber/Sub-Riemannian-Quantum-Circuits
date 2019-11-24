import functools
import unittest
from collections import Counter
from math import factorial
from unittest import TestCase
from subriemannian_qc.util import combinations, generate_allowed_subset
import numpy.testing as nptest
import numpy as np
import itertools
import random


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

class TestPermutations(TestCase):

    def test_num_distinct_permutations(self):
        # Generate 1000 lists
        for x in range(1000):
            # Each list consists of 10 elements uniformly randomly selected from {0, 1, ..., 5}
            elems = [random.randint(0, 5) for _ in range(10)]
            size = len(elems)
            cnt = Counter(elems)
            frequencies = cnt.items()
            redundancies = functools.reduce(lambda a, b: a * b, [factorial(value) for _, value in frequencies])
            num_perms = factorial(size) // redundancies
            self.assertEqual(num_perms, len(distinct_permutations(elems)))


class TestPauli(TestCase):

    def test_factorized(self):
        pass


class TestAllowed(TestCase):

    def test_allowed_size(self):
        for n in range(1, 20):
            # Count the number of one/two-body terms in the allowed set
            n_allowed = 9 * n * (n - 1) // 2 + 3 * n
            self.assertEqual(n_allowed, len(generate_allowed_subset(n)), 'Failure when n = {}'.format(n))
