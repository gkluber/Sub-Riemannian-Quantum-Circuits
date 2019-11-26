import functools
import warnings
from typing import List, Set, Tuple
import numpy as np
from qiskit.quantum_info.operators import Operator
from itertools import permutations
from subriemannian_qc.matrix_util import conj_transpose
from collections import Counter
from math import factorial

from subriemannian_qc.validate import is_matrix

I = np.identity(2, dtype=np.complex_)
X = np.array([[0, 1], [1, 0]], dtype=np.complex_)
Y = np.array([[0, 1j], [-1j, 0]], dtype=np.complex_)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex_)


def generate_lie_algebra(n: int):
    basis_indices = [0, 1, 2, 3]
    basis_combs = combinations(basis_indices, n)[1::]  # Exclude I since it's not in su(2^n)
    normalization = np.exp2(n / 2)

    # multiply by i to make them skew-Hermitian and factor in normalization
    basis = np.array([1j * get_pauli_tensor(string) / normalization for string in basis_combs])
    return basis


def generate_allowed_subset(n: int) -> np.ndarray:
    normalization = np.exp2(n / 2)
    if n == 0:
        return np.array([])
    elif n == 1:
        return 1j * np.array([I, X, Y, Z]) / normalization

    allowed_strings = [[0]*n]  # Include the identity to get U(2^n)

    # Generate all one-body basis elements
    for i in range(1, 4):
        for x in range(n):
            one_body = [0] * n
            one_body[x] = i
            allowed_strings.append(one_body)

    # Generate all two-body basis elements
    for i in range(1, 4):
        for j in range(i, 4):
            for x in range(n):
                for y in range(n if i != j else x):
                    if x == y:
                        continue
                    two_body = [0] * n
                    two_body[x] = i
                    two_body[y] = j
                    allowed_strings.append(two_body)

    # Convert index-strings to matrices. TODO compress them until actually needed for calculation
    allowed = []
    for string in allowed_strings:
        allowed.append(get_pauli_tensor(string))

    return 1j * np.array(allowed) / normalization


def get_pauli_tensor(index_string) -> np.ndarray:
    size = len(index_string)
    if size == 0:
        return np.identity(2)

    result = get_pauli(index_string[0])
    for x in range(1, size):
        result = np.kron(get_pauli(index_string[x]), result)
    return result


def get_pauli(index: int) -> np.ndarray:
    if index == 0:
        return np.identity(2)
    elif index == 1:
        return X
    elif index == 2:
        return Y
    elif index == 3:
        return Z


# Computes the linear combination of a vector and a list of matrices,
# such that the elements of the vectors become the weightings of the
# matrix linear combination
def vec_matrix_linear_comb(vector: np.ndarray, matrix_list: np.ndarray):
    return np.tensordot(vector, matrix_list, axes=1)


# Pretty ugly, but efficient. TODO refactor?
def combinations(elems: List[int], length: int):
    if len(elems) == 0 or length == 0:
        return []

    return np.stack(np.meshgrid(*tuple(elems for _ in range(length))), -1).reshape(-1, length)


# TODO maybe this will be needed?
def lazy_combinations(elems: List[int], length: int):
    pass


'''
def swap(arr: List[int], i: int, j: int):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp


# Iterative implementation of Heap's algorithm optimized for the case
# when there are many repeated elements
def distinct_permutations(elems: List[int]):
    results = []
    size = len(elems)
    elems = elems.copy()  # Don't want our swaps bleeding back into the original list
    if size == 0:
        return results

    # Compute the number of iterations we need to take
    cnt = Counter(elems)
    frequencies = cnt.items()
    redundancies = functools.reduce(lambda a,b: a*b, [factorial(value) for _, value in frequencies])
    num_perms = factorial(size) // redundancies

    results = set()
    _distinct_permutations_helper(elems, results, size)
    return results


# Permuting only elements in the slice elems[:k] at a time recursively
# This corresponds to a recursive implementation of Heap's algorithm, but
# we optimize for
def _distinct_permutations_helper(elems: List[int], results: Set[Tuple[int]], k: int):
    if k == 1:
        results.add(tuple(elems.copy()))
        return
    else:
        # Permute with element k - 1 unchanged
        _distinct_permutations_helper(elems, results, k - 1)
        for x in range(k):
            if k % 2 == 0:
                # If these two elements are the same, don't recurse
                if elems[x] == elems[k - 1]:
                    pass
                swap(elems, x, k - 1)
            else:
                # If these two elements are the same, don't recurse
                if elems[0] == elems[k - 1]:
                    pass
                swap(elems, 0, k - 1)
            _distinct_permutations_helper(elems, results, k - 1)
'''


def trace_error(a: np.ndarray, b: np.ndarray):
    if not is_matrix(a) or not is_matrix(b):
        raise ValueError

    n = a.shape[0]
    return 2 ** n - np.real(np.trace(np.matmul(a, conj_transpose(b))))


def elementwise_error(a: np.ndarray, b: np.ndarray):
    return np.sum(np.absolute(a - b) ** 2)
