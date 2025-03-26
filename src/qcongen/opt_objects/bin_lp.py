from math import ceil, abs

import numpy as np
import logging

from qcongen.opt_objects.ising import IsingHamiltonian

logger = logging.getLogger('qcongen')

class BLP:
    """Binary Linear Programming model class.

    Contains the components of a binary linear programming problem:
    min c^T x
    s.t. Ax = b
         x binary

    Also maintains auxiliary matrices Ahat and bhat for problem modifications.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        """Initialize BLP with problem matrices and vectors.

        Args:
            A: 2D constraint coefficient matrix of shape (m,n)
            b: 1D right-hand side vector of shape (m,)
            c: 1D objective coefficient vector of shape (n,)

        Raises:
            ValueError: If dimensions are inconsistent or inputs invalid
        """
        # Validate inputs
        if len(A.shape) != 2:
            raise ValueError("A must be a 2D matrix")
        if len(b.shape) != 1 or len(c.shape) != 1:
            raise ValueError("b and c must be 1D vectors")
        if A.shape[0] != b.shape[0]:
            raise ValueError("A and b have inconsistent dimensions")
        if A.shape[1] != c.shape[0]:
            raise ValueError("A and c have inconsistent dimensions")

        self.A: np.ndarray = A.astype(float)
        self.b: np.ndarray = b.astype(float)
        self.c: np.ndarray = c.astype(float)
        self.active_constraints: np.ndarray = np.zeros(A.shape[0])

        self.Ahat: np.ndarray = np.zeros_like(A)
        self.bhat: np.ndarray = np.zeros_like(b)

        self.x: np.ndarray = np.zeros_like(c)

        self.M = self._calculate_M()

    def copy(self) -> 'BLP':
        """Create a deep copy of the BLP instance.
        
        Returns:
            BLP: A new BLP instance with copies of all data
        """
        new_blp = BLP(
            A=self.A.copy(),
            b=self.b.copy(),
            c=self.c.copy(),
        )
        new_blp.active_constraints = self.active_constraints.copy()
        new_blp.Ahat = self.Ahat.copy()
        new_blp.bhat = self.bhat.copy()
        new_blp.x = self.x.copy()
        new_blp.M = self.M
        return new_blp

    @property
    def n(self) -> int:
        """Number of variables in the problem."""
        return self.A.shape[1]

    @property
    def m(self) -> int:
        """Number of constraints in the problem."""
        return self.A.shape[0]

    def _calculate_M(self) -> float:
        """Calculate the penalty constant M for the Ising conversion.

        Returns:
            float: The penalty constant M
        """
        kappa = 1.0  # coeff matrix and b are integers
        return ceil(sum(abs(c) for c in self.c) * kappa)

    @property
    def is_complete(self) -> bool:
        """Check if the BLP is complete.

        Returns:
            bool: True if the BLP is complete, False otherwise
        """
        return bool(np.all(self.active_constraints))

    def _add_constraint(self, cstr_index: int) -> None:
        """Add a constraint to the BLP.

        Args:
            cstr_index: The index of the constraint to add
        """
        self.active_constraints[cstr_index] = 1
        self.Ahat[cstr_index, :] = self.A[cstr_index, :]
        self.bhat[cstr_index] = self.b[cstr_index]

    def toMPS(self, path: str) -> None:
        """Write the linear program to an MPS format file.

        The problem is in the form:
        min c^T x
        s.t. Ax = b
             x binary

        MPS format follows IBM standard:
        - Fixed column format
        - Each section starts with a header
        - ROWS: N for objective, E for equality constraints
        - COLUMNS: Variable coefficients
        - RHS: Right-hand side values
        - BOUNDS: Variable bounds (BV for binary)

        Args:
            path: File path to write the MPS file

        Raises:
            OSError: If path is invalid or write fails
        """
        m, n = self.A.shape

        try:
            with open(path, "w") as f:
                f.write("NAME          SET_PARTITION\n")

                f.write("ROWS\n")
                f.write(" N  COST\n")
                for i in range(m):
                    f.write(f" E  ELEM{i+1}\n")

                f.write("COLUMNS\n")
                for j in range(n):
                    if self.c[j] != 0:
                        f.write(f"    X{j+1:<8}COST      {self.c[j]:<8.0f}")
                    
                    nonzero_rows = np.nonzero(self.A[:, j])[0]
                    if len(nonzero_rows) > 0:
                        if self.c[j] != 0:
                            f.write("   ")
                        
                        for idx, i in enumerate(nonzero_rows):
                            if idx > 0 and idx % 2 == 0:
                                f.write("\n    X{:<8}".format(j+1))
                            f.write(f"ELEM{i+1:<6}{self.A[i,j]:<8.0f}   ")
                    f.write("\n")

                f.write("RHS\n")
                for i in range(m):
                    if self.b[i] != 0:
                        f.write(f"    RHS       ELEM{i+1:<6}{self.b[i]:<8.0f}\n")

                f.write("BOUNDS\n")
                for j in range(n):
                    f.write(f" BV BND1      X{j+1}\n")

                f.write("ENDATA\n")

        except OSError as e:
            raise OSError(f"Failed to write MPS file to {path}: {str(e)}") from e

    def to_ising(self) -> IsingHamiltonian:
        """Convert the BLP to an IsingHamiltonian.

        Converts from {0,1} binary variables to {-1,1} spin variables.
        Uses the formulation:
        J = -1/4 * M * Ahat.T @ Ahat
        h = c.T - 2M * bhat.T @ Ahat + M * ones.T @ (Ahat.T @ Ahat)
        μ = -1/2
        const = 1/4 * M * ones.T @ (Ahat.T @ Ahat) @ ones + 1/2 * ones.T @ c
                + M * bhat.T @ Ahat @ ones + M * bhat.T @ bhat
        where M is a large penalty constant

        Returns:
            IsingHamiltonian: The converted Ising model
        """
        _, n = self.A.shape

        ones = np.ones(n)

        AhatT_Ahat = self.Ahat.T @ self.Ahat
        bhat_Ahat = self.bhat.T @ self.Ahat

        J = -0.25 * self.M * AhatT_Ahat

        h = self.c - (2 * self.M * bhat_Ahat + self.M * ones @ AhatT_Ahat)

        const = (
            0.25 * self.M * ones @ AhatT_Ahat @ ones
            + 0.5 * ones @ self.c
            + self.M * bhat_Ahat @ ones
            + self.M * (self.bhat @ self.bhat)
        )

        return IsingHamiltonian(J, h, -0.5, const)

    def check_feasibility(self, X: np.ndarray) -> tuple[bool, list[int], float]:
        """Check feasibility of solution candidates.

        Args:
            X: Matrix where each column is a candidate solution vector

        Returns:
            tuple containing:
                bool: True if feasible solution found, False otherwise
                list[int]: Best feasible solution if found, empty list otherwise
                float: Objective value of best solution if found, INF otherwise
        """
        INF = 1e8  # Define infinity constant

        n = self.A.shape[1]
        if X.shape[0] != n:
            raise ValueError(f"Solution vectors must have dimension {n}")

        best_sol: list[int] = []
        best_val: float = INF
        found_feasible = False

        for j in range(X.shape[1]):
            x = X[:, j]

            # Check if solution satisfies active constraints
            residuals = np.abs(self.A @ x - self.b)
            
            is_feasible = np.all(residuals < 1e-10)

            if is_feasible:
                obj_val = float(self.c @ x)
                logger.info(f"Found feasible solution: {x} with objective value {obj_val}")
                if obj_val < best_val:
                    best_val = obj_val
                    best_sol = x.astype(int).tolist()
                    found_feasible = True

        return found_feasible, best_sol, best_val

    def get_violation_scores(self, X: np.ndarray, total_sample_size: int) -> np.ndarray:
        """Calculate violation scores for all constraints.
        
        Args:
            X: Matrix where each column is a candidate solution vector
            total_sample_size: Total number of samples (sum of all sample counts)
            
        Returns:
            np.ndarray: Vector of violation scores for each constraint
        """
        # Compute AX - b for all samples at once
        residuals = self.A @ X - self.b[:, np.newaxis]

        V = (np.abs(residuals) > 1e-10).astype(float)

        nu = np.sum(V, axis=1) / total_sample_size

        return nu

    def add_constraints(self, nu: np.ndarray, t: float = 0.0) -> list[int]:
        """Choose the constraints to add to the BLP.

        Args:
            X: Matrix where each column is a candidate solution vector, shape (n,s)
            nu: Vector of violation scores, shape (m,)
        """
        new_cstrs: list[int] = np.where((nu >= t) & (self.active_constraints == 0))[0].tolist()
        for cstr in new_cstrs:
            self._add_constraint(cstr)
        return new_cstrs

