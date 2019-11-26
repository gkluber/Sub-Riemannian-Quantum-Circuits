from typing import List

import numpy as np

from subriemannian_qc.matrix_util import conj_transpose
from subriemannian_qc.validate import is_matrix

'''
The common computational tasks needed by all of the geodesic methods.
'''


I = np.identity(2, dtype=np.complex_)
X = np.array([[0, 1], [1, 0]], dtype=np.complex_)
Y = np.array([[0, 1j], [-1j, 0]], dtype=np.complex_)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex_)


# Generates the Lie algebra u(2^n)
def generate_lie_algebra(n: int):
    basis_indices = [0, 1, 2, 3]
    basis_combs = combinations(basis_indices, n)
    normalization = np.exp2(n / 2)

    # multiply by i to make them skew-Hermitian and factor in normalization
    basis = np.array([1j * get_pauli_tensor(string) / normalization for string in basis_combs])
    return basis


# Generates the allowed subset for u(2^n)
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


# Given a tuple in the form (a_1, a_2, ..., a_n), where a_i is in [0,1,2,3]
# Outputs the corresponding tensor product, such that the first element in the tuple
# corresponds to the least significant (right-most) factor of the tensor product
def get_pauli_tensor(index_string) -> np.ndarray:
    size = len(index_string)
    if size == 0:
        return np.identity(2)

    result = get_pauli(index_string[0])
    for x in range(1, size):
        result = np.kron(get_pauli(index_string[x]), result)
    return result


# Convert number (0,1,2,3) to 2x2 complex Pauli matrix
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
# matrix linear combination. For example, we could use this to decompose
# a 2x2 matrix A as aI + bX + cY + dZ by passing in np.array([a,b,c,d])
# and np.array([I,X,Y,Z])
def vec_matrix_linear_comb(vector: np.ndarray, matrix_list: np.ndarray):
    return np.tensordot(vector, matrix_list, axes=1)


# Pretty ugly, but efficient way of generating combinations with replacement.
def combinations(elems: List[int], length: int):
    if len(elems) == 0 or length == 0:
        return []

    return np.stack(np.meshgrid(*tuple(elems for _ in range(length))), -1).reshape(-1, length)


# Compute the trace error of two matrices. Note that this may be negative,
# but is guaranteed to be symmetric.
def trace_error(a: np.ndarray, b: np.ndarray):
    if not is_matrix(a) or not is_matrix(b):
        raise ValueError

    # TODO further validate inputs

    n = a.shape[0]
    return 2 ** n - np.real(np.trace(np.matmul(a, conj_transpose(b))))


# Simply sum the magnitudes of the differences for each slot in the matrix
def elementwise_error(a: np.ndarray, b: np.ndarray):
    # TODO validate inputs
    return np.sum(np.absolute(a - b) ** 2)
