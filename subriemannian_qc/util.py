from typing import List
import numpy as np


def generate_lie_group(n: int):
    basis = [1, 2, 3, 4]
    np.meshgrid(basis, basis, basis, basis).T.reshape(-1, )


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
