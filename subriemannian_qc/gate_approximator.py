from abc import ABC, abstractmethod

import numpy as np

'''
Abstract base class that defines the interface for all geodesic gate approximators.
'''


class GateApproximator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def approx_matrix(self, unitary: np.ndarray):
        pass
