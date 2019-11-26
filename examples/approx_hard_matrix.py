from scipy.stats import unitary_group

from subriemannian_qc.discrete import DiscreteApproximator
from subriemannian_qc.util import elementwise_error

#  Approximates the following "hard" matrix.
for _ in range(5):
    #  By the pigeonhole principle, we know that the vast majority of matrices are hard to evaluate
    #  With overwhelming probability, this matrix will be hard:
    target = unitary_group.rvs(8)
    gate_approx = DiscreteApproximator(3, 10, 10, 1, 100)
    approx, _ = gate_approx.approx_matrix(target)

    print(target)
    print(approx)

    print('Error: {}'.format(elementwise_error(approx, target)))
