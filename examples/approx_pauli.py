from subriemannian_qc.discrete import DiscreteApproximator
from subriemannian_qc.util import get_pauli, elementwise_error
import numpy as np


#  Use the discrete approximator to find approximations to the Pauli matrices
for p in range(0, 4):
    gate_approx = DiscreteApproximator(1, 10, 10, 1, 100)
    approximation = gate_approx.approx_matrix(get_pauli(p))
    print(approximation)
    print('Error: {}'.format(elementwise_error(approximation, get_pauli(p))))
