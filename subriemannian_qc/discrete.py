from functools import partial, partialmethod
import qiskit
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from subriemannian_qc.validate import is_unitary_matrix
from subriemannian_qc.gate_approximator import GateApproximator
from subriemannian_qc.util import generate_lie_algebra, vec_matrix_linear_comb, generate_allowed_subset, trace_error, \
    elementwise_error


class DiscreteApproximator(GateApproximator):
    def __init__(self, n: int, num_segments: int, p: int, q: int, omega: float):
        super().__init__()

        # Validate inputs
        if n <= 0 or num_segments <= 0:
            raise ValueError

        self.n = n  # The number of qubits in the space
        self.N = 2 ** n  # The dimensionality of the space
        self.num_segments = num_segments  # The number of segments in the discrete geodesics approximation
        self.allowed = generate_allowed_subset(self.n)  # Precompute the allowed set basis
        self.n_allowed = len(self.allowed)  # Precompute the size of the allowed set

        # Parameters for the cost function
        self.p = p
        self.q = q
        self.omega = omega

    def approx_matrix(self, unitary: np.ndarray):
        if not is_unitary_matrix(unitary) or unitary.shape[0] != self.N:
            return None

        coefficients = np.random.rand(self.num_segments * self.n_allowed) / 10
        cost = partial(self._cost, target=unitary, omega=self.omega, p=self.p, q=self.q, num_segments=self.num_segments,
                       allowed=self.allowed)
        res = minimize(cost, coefficients, options={'disp': True})
        optimal_coefficients = res.x
        return self._get_matrix(optimal_coefficients, self.num_segments, self.allowed)

    @staticmethod
    def _get_matrix(coefficients: np.ndarray, num_segments: int, allowed: np.ndarray):
        n_allowed = len(allowed)
        if len(coefficients) != num_segments * n_allowed:
            return None

        x = expm(vec_matrix_linear_comb(coefficients[0:n_allowed], allowed))
        for y in range(1, num_segments):
            # Get the Lie algebra member for this segment
            seg = vec_matrix_linear_comb(coefficients[y * n_allowed:(y + 1) * n_allowed], allowed)
            # Multiply its exponential into the product
            x = np.matmul(expm(seg), x)

        return x

    @staticmethod
    def _cost(coefficients: np.ndarray, target: np.ndarray, omega: int, p: int, q: int, num_segments: int,
              allowed: np.ndarray):
        x = DiscreteApproximator._get_matrix(coefficients, num_segments, allowed)
        error = elementwise_error(x, target)
        energy = DiscreteApproximator._get_energy(coefficients)
        return omega * error + (10 ** (-p) * error + 10 ** (-q)) * energy

    def _gradient_error(self, coefficients: np.ndarray, target: np.ndarray):
        pass  # TODO

    def _gradient_energy(self, coefficients: np.ndarray):
        pass  # TODO

    def _gradient_cost(self, coefficients: np.ndarray, target: np.ndarray):
        pass  # TODO

    @staticmethod
    def _get_energy(a: np.ndarray):
        # Element-wise product of a and its conjugate transpose. Sum it all together
        return np.sum(a * np.conj(a))
