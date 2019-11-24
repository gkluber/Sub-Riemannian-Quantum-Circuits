import qiskit
import numpy as np
from scipy.linalg import expm

from subriemannian_qc.validate import is_unitary_matrix
from subriemannian_qc.gate_approximator import GateApproximator
from subriemannian_qc.util import generate_lie_algebra, vec_matrix_linear_comb


class DiscreteApproximator(GateApproximator):
    def __init__(self, num_segments: int):
        super().__init__()
        self.num_segments = num_segments

    def approx_matrix(self, unitary: np.ndarray):
        if not is_unitary_matrix(unitary):
            return None

        # Overall dimension
        n = unitary.shape[0]
        # Dimension of the allowed sub-Riemannian space
        n_allowed = 9 * n * (n - 1) // 2 + 3 * n

        su = generate_lie_algebra(n)

        coefficients = np.zeros(self.num_segments * n_allowed)

        x = expm(vec_matrix_linear_comb(coefficients[:n_allowed], su[:n_allowed]))


    @staticmethod
    def get_matrix():
        pass

    @staticmethod
    def get_energy(a: np.ndarray):
        # Element-wise product of a and its conjugate transpose. Sum it all together
        return np.sum(a * np.conj(a))
