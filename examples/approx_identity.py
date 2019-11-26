from subriemannian_qc.discrete import DiscreteApproximator
import numpy as np


#  Use the discrete approximator to find an approximation to the identity matrix
for n in range(1, 4):
    gate_approx = DiscreteApproximator(n, 10, 10, 1, 100)
    approximation = gate_approx.approx_matrix(np.identity(2**n))
    print(approximation)
