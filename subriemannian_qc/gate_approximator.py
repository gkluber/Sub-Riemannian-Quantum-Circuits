from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit
import numpy as np


class GateApproximator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def approx_matrix(self, unitary: np.ndarray):
        pass
