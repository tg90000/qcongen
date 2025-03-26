"""Constraint generation algorithm implementation."""

import logging
import numpy as np
from pathlib import Path

from qcongen.opt_objects.bin_lp import BLP
from qcongen.opt_objects.ising import IsingHamiltonian
from qcongen.opt_objects.quantum_problem_qiskit import QuantumProblem, SimulatedQuantumProblem
from qcongen.utils.logging import setup_logging
from qcongen.utils.reference_partition import solve_blp_with_ortools

# Get the package logger
logger = logging.getLogger('qcongen')

def run_constraint_gen(
    blp: BLP,
    sample_size: int = 1000,  # Total number of shots/measurements
    t: float = 0.0,
    max_iters: int = 1000,
    log_dir: Path | None = None,
) -> tuple[bool, list[int], float]:
    """Run the constraint generation algorithm.

    Implements Algorithm 1 from the paper:
    While there exists all-zero row in Ahat:
        1. Calculate Hamiltonian H from (c, Ahat, bhat)
        2. Get quantum samples X from H using QAOA:
           a. Create and optimize QAOA circuit for H
           b. Sample from optimized circuit with sample_size shots
           c. Get unique samples with their probabilities
        3. Check feasibility of samples
        4. If feasible solution found, return it
        5. Calculate violation scores
        6. Add constraints based on scores

    Args:
        blp: The BLP to run the constraint generation algorithm on
        sample_size: Total number of quantum measurements (shots) to take
        t: Threshold for adding constraints
        max_iters: Maximum number of iterations
        log_dir: Directory to store log files (optional, defaults to timestamped dir in /results)

    Returns:
        tuple containing:
            bool: True if feasible solution found
            list[int]: The solution if found, empty list otherwise
            float: The final objective value if found, INF otherwise
    """
    log_dir = setup_logging(log_dir)

    logger.info("\nProblem information:")
    logger.info(f"  Number of variables (n): {blp.n}")
    logger.info(f"  Number of constraints (m): {blp.m}")
    logger.debug(f"  Density of A: {np.count_nonzero(blp.A)/(blp.m*blp.n):.2%}")

    # Run classical reference solver first
    logger.info("\nRunning classical reference solver:")
    ref_success, ref_solution, ref_value = solve_blp_with_ortools(blp)
    if ref_success:
        logger.info("  Reference solution found:")
        logger.info(f"    Solution: {ref_solution}")
        logger.info(f"    Value: {ref_value}")
    else:
        logger.info("  Reference solver found no feasible solution")

    logger.info("\nStarting quantum constraint generation with parameters:")
    logger.info(f"  sample_size (total shots): {sample_size}")
    logger.info(f"  t: {t}")
    logger.info(f"  max_iters: {max_iters}")

    iteration = 0
    t_multiplier = 1.0
    init_params = np.random.random(4)  # 2 layers (p=2) -> 4 parameters (2 gamma, 2 beta)
    while iteration < max_iters:  # while exists all-zero row in Ahat
        logger.info(f"\nIteration {iteration}:")

        # Calculate Hamiltonian H from (c, Ahat, bhat)
        H: IsingHamiltonian = blp.to_ising()
        logger.debug("Generated Ising Hamiltonian")

        # Create quantum problem instance for QAOA
        qp = QuantumProblem(
            hamiltonian=H,
            sample_size=sample_size,
            init_params=init_params,
        )
        
        logger.debug("Optimizing QAOA circuit...")
        result = qp.optimize_circuit()
        
        if result.success:
            logger.debug(f"Circuit optimization successful, final value: {result.fun}")
        else:
            logger.warning("Circuit optimization did not converge")

        # Sample from the optimized circuit
        logger.debug("Sampling from optimized circuit...")
        samples_dict = qp.sample_circuit()  # Returns dict[int, tuple[np.ndarray, int]]
        
        # Extract unique samples and their counts
        X = np.column_stack([sample_tuple[0] for sample_tuple in samples_dict.values()])  # Each column is a unique sample
        sample_counts = np.array([sample_tuple[1] for sample_tuple in samples_dict.values()])
        total_samples = np.sum(sample_counts)
        
        n_unique_samples = X.shape[1]
        logger.debug(f"Generated {n_unique_samples} unique samples from {total_samples} total measurements")

        # Check feasibility of samples
        is_feasible, solution, value = blp.check_feasibility(X) 
        if is_feasible:
            logger.info("  Found feasible solution:")
            logger.info(f"    Solution: {solution}")
            logger.info(f"    Value: {value}")
            gap = ((value - ref_value) / ref_value) * 100 if ref_value != 0 else float('inf')
            logger.info(f"    Gap to reference: {gap:.2f}%")
            return True, solution, value  # Return x if feasible

        logger.info("  No feasible solution found in this iteration")
        if blp.is_complete:
            logger.info("  All constraints added, exiting")
            return False, [], float("inf")

        # Calculate violation scores and add constraints
        nu = blp.get_violation_scores(X, total_samples)  
        t_multiplier = 1.0
        
        while True:
            t = t_multiplier * max(nu)
            new_constraints = blp.add_constraints(nu, t)  
            
            if new_constraints:
                logger.info(f"  Added {len(new_constraints)} new constraints: {new_constraints}")
                break
                
            old_t_multiplier = t_multiplier
            t_multiplier = max(0.0, t_multiplier - 0.1)
            if t_multiplier == 0.0:
                logger.info(f"  No new constraints added, t multiplier reached zero")
                break
                
            logger.info(f"  No new constraints added, lowered t multiplier: {old_t_multiplier:.1f} -> {t_multiplier:.1f}")
        
        init_params = result.x  
        iteration += 1

    logger.info("\nConstraint generation completed:")
    logger.info(f"  Total iterations: {iteration}")
    logger.info("  No feasible solution found")
    if ref_success:
        logger.info("\nBest known solution (from reference solver):")
        logger.info(f"  Solution: {ref_solution}")
        logger.info(f"  Value: {ref_value}")

    return False, [], float("inf")
