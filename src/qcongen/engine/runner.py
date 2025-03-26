"""Runner module for QConGen execution logic."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from qcongen.engine.constraint_gen import run_constraint_gen
from qcongen.engine.ref_qaoa import run_reference_qaoa
from qcongen.io.config_reader import QConGenConfig
from qcongen.io.input_reader import MPS_to_BLP
from qcongen.io.output_writer import create_output_directory
from qcongen.utils.reference_partition import solve_mps_with_ortools
from qcongen.utils.generators import generate_set_partition_instance
from qcongen.opt_objects.bin_lp import BLP

logger = logging.getLogger('qcongen')

SolverType = Literal["classical", "qaoa", "constraint_gen"]

def run_classical_solver(
    blp: BLP,
    instance_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[bool, list[int], float]:
    """Run classical reference solver (SCIP).
    
    Args:
        blp: Binary Linear Program instance
        instance_path: Path to MPS file (if available)
        output_dir: Directory for output files
        
    Returns:
        tuple of (success, solution, value)
    """
    logger.info("\nRunning Classical Reference Solver (SCIP):")
    if instance_path is not None:
        success, solution, value = solve_mps_with_ortools(instance_path)
    else:
        success, solution, value = solve_mps_with_ortools(blp)
        
    if success:
        logger.info("  Reference solution found:")
        logger.info(f"    Solution: {solution}")
        logger.info(f"    Value: {value}")
    else:
        logger.info("  Reference solver found no feasible solution")
        
    return success, solution, value

def run_qaoa_solver(
    blp: BLP,
    sample_size: int,
    output_dir: Path | None = None,
) -> tuple[bool, list[int], float]:
    """Run QAOA reference solution.
    
    Args:
        blp: Binary Linear Program instance
        sample_size: Number of quantum measurements
        output_dir: Directory for output files
        
    Returns:
        tuple of (success, solution, value)
    """
    logger.info("\nRunning QAOA Reference Solution (with all constraints):")
    success, solution, value = run_reference_qaoa(
        blp=blp.copy(),
        sample_size=sample_size,
        log_dir=output_dir,
    )
    
    if success:
        logger.info("  QAOA solution found:")
        logger.info(f"    Solution: {solution}")
        logger.info(f"    Value: {value}")
    else:
        logger.info("  QAOA solver found no feasible solution")
        
    return success, solution, value

def run_constraint_gen_solver(
    blp: BLP,
    sample_size: int,
    t: float,
    max_iters: int,
    output_dir: Path | None = None,
) -> tuple[bool, list[int], float]:
    """Run constraint generation algorithm.
    
    Args:
        blp: Binary Linear Program instance
        sample_size: Number of quantum measurements
        t: Threshold for adding constraints
        max_iters: Maximum number of iterations
        output_dir: Directory for output files
        
    Returns:
        tuple of (success, solution, value)
    """
    logger.info("\nRunning Constraint Generation:")
    success, solution, value = run_constraint_gen(
        blp=blp,
        sample_size=sample_size,
        t=t,
        max_iters=max_iters,
        log_dir=output_dir,
    )
    
    if success:
        logger.info("  Constraint generation solution found:")
        logger.info(f"    Solution: {solution}")
        logger.info(f"    Value: {value}")
    else:
        logger.info("  Constraint generation found no feasible solution")
        
    return success, solution, value

def run_solver(
    solver_type: SolverType,
    blp: BLP,
    config: QConGenConfig,
    t: float = 0.0,
    max_iters: int = 1000,
    instance_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[bool, list[int], float]:
    """Run the specified solver on the given problem instance.
    
    Args:
        solver_type: Type of solver to use
        blp: Binary Linear Program instance
        config: Configuration for the run
        t: Threshold for adding constraints
        max_iters: Maximum number of iterations
        instance_path: Path to MPS file (if available)
        output_dir: Directory for output files
        
    Returns:
        tuple of (success, solution, value)
    """
    if solver_type == "classical":
        return run_classical_solver(blp, instance_path, output_dir)
    elif solver_type == "qaoa":
        return run_qaoa_solver(blp, config.sample_size, output_dir)
    elif solver_type == "constraint_gen":
        return run_constraint_gen_solver(blp, config.sample_size, t, max_iters, output_dir)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

def run_single_instance(
    config: QConGenConfig,
    t: float,
    max_iters: int,
    use_ref: bool,
    compare_ref: bool,
    output_dir: Path | None = None,
) -> tuple[bool, list[int], float]:
    """Run QConGen on a single instance.
    
    Args:
        config: Configuration for the run
        t: Threshold for adding constraints
        max_iters: Maximum number of iterations
        use_ref: Whether to use classical reference solver only
        compare_ref: Whether to compare with QAOA reference solution
        output_dir: Directory for output files (created if None)
        
    Returns:
        tuple of (success, solution, value)
    """
    if output_dir is None:
        output_dir = create_output_directory()

    if config.input_type == "mps":
        blp = MPS_to_BLP(config.input_file_path_resolved)
        instance_path = config.input_file_path_resolved
    elif config.input_type == "random":
        if config.random_instance is None:
            blp = generate_set_partition_instance()
            instance_path = None
        else:
            blp = generate_set_partition_instance(
                n_sets=config.random_instance.n_sets,
                n_elements=config.random_instance.n_elements,
                min_set_size=config.random_instance.min_set_size,
                max_set_size=config.random_instance.max_set_size,
                min_cost=config.random_instance.min_cost,
                max_cost=config.random_instance.max_cost,
                instance_name=config.random_instance.instance_name,
                output_dir=output_dir,
            )
            instance_path = output_dir / f"{config.random_instance.instance_name}.mps"
    else:
        raise ValueError(f"Unsupported input type: {config.input_type}")

    logger.info("\nProblem information:")
    logger.info(f"  Number of variables (n): {blp.n}")
    logger.info(f"  Number of constraints (m): {blp.m}")
    logger.debug(f"  Density of A: {np.count_nonzero(blp.A)/(blp.m*blp.n):.2%}")

    if use_ref:
        return run_solver("classical", blp, config, instance_path=instance_path, output_dir=output_dir)

    qaoa_success = False
    qaoa_value = float('inf')
    if compare_ref:
        qaoa_success, _, qaoa_value = run_solver("qaoa", blp, config, output_dir=output_dir)

    ref_success, _, ref_value = run_solver("classical", blp, config, instance_path=instance_path, output_dir=output_dir)

    success, solution, value = run_solver(
        "constraint_gen",
        blp,
        config,
        t=t,
        max_iters=max_iters,
        output_dir=output_dir,
    )

    if success and (ref_success or qaoa_success):
        logger.info("\n---")
        logger.info("Solution Comparison:")
        if ref_success:
            gap_ref = ((value - ref_value) / ref_value) * 100 if ref_value != 0 else float('inf')
            logger.info(f"Gap to classical reference: {gap_ref:.2f}%")
        if qaoa_success:
            gap_qaoa = ((value - qaoa_value) / qaoa_value) * 100 if qaoa_value != 0 else float('inf')
            logger.info(f"Gap to QAOA reference: {gap_qaoa:.2f}%")

    return success, solution, value 