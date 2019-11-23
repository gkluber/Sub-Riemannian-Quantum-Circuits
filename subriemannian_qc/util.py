from typing import List
import numpy as np

X = np.array([0, 1], [1, 0], dtype=np.complex64)
Y = np.array([0, 1j], [-1j, 0], dtype=np.complex64)
Z = np.array([1, 0], [0, -1], dtype=np.complex64)


def generate_lie_group(n: int):
    basis_indices = [0, 1, 2, 3]

    # Need to exclude the [0, 0, 0, 0] identity matrix
    basis_combs = combinations(basis_indices, n)[1:]


def get_pauli_tensor(index_string: List[int]) -> np.ndarray:
    size = len(index_string)
    if size == 0:
        return np.identity(2)

    result = get_pauli(index_string[0])
    for x in range(1, size):
        result = np.kron(result, get_pauli(index_string[x]))
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


def generate_lie_algebra(n: int):
    pass


# Pretty ugly, but efficient. TODO refactor?
def combinations(elems: List[int], length: int):
    if len(elems) == 0 or length == 0:
        return []

    return np.stack(np.meshgrid(*tuple(elems for _ in range(length))), -1).reshape(-1, length)


# TODO maybe this will be needed?
def lazy_combinations(elems: List[int], length: int):
    pass
