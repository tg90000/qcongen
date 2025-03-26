import numpy as np


class IsingHamiltonian:
    """Class representing an Ising Hamiltonian.

    The Hamiltonian is of the form:
    H(sigma) = -∑(i,j) J[i,j]sigma[i]sigma[j] - μ∑(i) h[i]sigma[i]
    where sigma[i] ∈ {-1,1} are spin variables
    """

    def __init__(self, J: np.ndarray, h: np.ndarray, mu: float, const: float):
        """Initialize Ising Hamiltonian.

        Args:
            J: 2D coupling matrix (symmetric)
            h: 1D local field vector
            mu: Field strength multiplier
            const: addition factor upon conversion to Ising model

        Raises:
            ValueError: If dimensions are inconsistent or inputs invalid
        """
        if len(J.shape) != 2:
            raise ValueError("J must be a 2D matrix")
        if J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square matrix")
        if len(h.shape) != 1:
            raise ValueError("h must be a 1D vector")
        if J.shape[0] != h.shape[0]:
            raise ValueError("J and h have inconsistent dimensions")

        self.J: np.ndarray = J.astype(float)
        self.h: np.ndarray = h.astype(float)
        self.mu: float = float(mu)
        self.constant: float = const
