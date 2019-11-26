from unittest import TestCase

import numpy as np
import numpy.testing as nptest

from subriemannian_qc.util import combinations, generate_allowed_subset, generate_lie_algebra

'''
Unit tests for utility functions. Due to the probabilistic nature of optimization, we didn't write
any unit tests for finding optimal gates and instead verified this using examples.
'''


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


class TestPauli(TestCase):

    def test_factorized(self):
        pass  # TODO test generate_pauli_tensor


class TestLieAlgebra(TestCase):

    def test_size(self):
        for n in range(1, 5):
            algebra = generate_lie_algebra(n)
            expected = 4**n
            actual = len(algebra)
            self.assertEqual(expected, actual, 'Failure when n = {}'.format(n))


class TestAllowed(TestCase):

    def test_allowed_size(self):
        for n in range(1, 10):
            # Count the number of one/two-body terms in the allowed set
            n_allowed = 9 * n * (n - 1) // 2 + 3 * n + 1
            self.assertEqual(n_allowed, len(generate_allowed_subset(n)), 'Failure when n = {}'.format(n))
