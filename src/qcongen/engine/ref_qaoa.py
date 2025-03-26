"""Reference QAOA implementation for comparison with constraint generation."""

import logging
import numpy as np
from pathlib import Path

from qcongen.opt_objects.bin_lp import BLP
from qcongen.opt_objects.ising import IsingHamiltonian
from qcongen.opt_objects.quantum_problem_qiskit import QuantumProblem
from qcongen.utils.logging import setup_logging

logger = logging.getLogger('qcongen')

def run_reference_qaoa(
    blp: BLP,
    sample_size: int = 1000, 
    log_dir: Path | None = None,
) -> tuple[bool, list[int], float]:
    """Run QAOA on the full problem with all constraints.
    
    This serves as a reference implementation to compare against the
    constraint generation approach. Instead of iteratively adding constraints,
    it includes all constraints from the start.

    Args:
        blp: The BLP to run QAOA on
        sample_size: Total number of quantum measurements (shots) to take
        log_dir: Directory to store log files (optional)

    Returns:
        tuple containing:
            bool: True if feasible solution found
            list[int]: The solution if found, empty list otherwise
            float: The objective value if found, INF otherwise
    """
    log_dir = setup_logging(log_dir)

    logger.info("\nProblem information:")
    logger.info(f"  Number of variables (n): {blp.n}")
    logger.info(f"  Number of constraints (m): {blp.m}")
    logger.debug(f"  Density of A: {np.count_nonzero(blp.A)/(blp.m*blp.n):.2%}")

    logger.info("\nStarting reference QAOA with parameters:")
    logger.info(f"  sample_size (total shots): {sample_size}")

    # Add all constraints at the start
    for i in range(blp.m):
        blp._add_constraint(i)
    logger.info("Added all constraints")

    # Calculate Hamiltonian H from full problem
    H: IsingHamiltonian = blp.to_ising()
    logger.debug("Generated Ising Hamiltonian")

    init_params = np.random.random(4) 
    qp = QuantumProblem(
        hamiltonian=H,
        sample_size=sample_size,
        init_params=init_params,
    )
    
    # Optimize the circuit
    logger.debug("Optimizing QAOA circuit...")
    result = qp.optimize_circuit()
    
    if result.success:
        logger.debug(f"Circuit optimization successful, final value: {result.fun}")
    else:
        logger.warning("Circuit optimization did not converge")

    # Sample from the optimized circuit
    logger.debug("Sampling from optimized circuit...")
    samples_dict = qp.sample_circuit()
    
    # Extract unique samples and their counts
    X = np.column_stack([sample_tuple[0] for sample_tuple in samples_dict.values()])
    sample_counts = np.array([sample_tuple[1] for sample_tuple in samples_dict.values()])
    total_samples = np.sum(sample_counts)
    
    n_unique_samples = X.shape[1]
    logger.debug(f"Generated {n_unique_samples} unique samples from {total_samples} total measurements")

    # Check feasibility of samples
    is_feasible, solution, value = blp.check_feasibility(X)
    if is_feasible:
        logger.info("Found feasible solution:")
        logger.info(f"  Solution: {solution}")
        logger.info(f"  Value: {value}")
        return True, solution, value

    logger.info("No feasible solution found")
    return False, [], float("inf") 