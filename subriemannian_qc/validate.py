import numpy as np
from subriemannian_qc.matrix_util import conj_transpose


def is_matrix(a: np.ndarray) -> bool:
    return len(a.shape) == 2 and a.shape[0] == a.shape[1]


# Not very efficient, but gets the job done
def is_unitary_matrix(a: np.ndarray) -> bool:
    if not is_matrix(a):
        return False

    return np.isclose(np.linalg.inv(a), conj_transpose(a)).all()
