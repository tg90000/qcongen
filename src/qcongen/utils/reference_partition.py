"""Reference solver using OR-Tools."""

from pathlib import Path
from typing import Tuple

import numpy as np
from ortools.linear_solver import pywraplp

from qcongen.io.input_reader import MPS_to_BLP
from qcongen.opt_objects.bin_lp import BLP


def solve_mps_with_ortools(path: Path | str) -> Tuple[bool, list[int], float]:
    """Solve an MPS file using OR-Tools.
    
    Args:
        path: Path to the MPS file
        
    Returns:
        tuple containing:
            bool: True if feasible solution found
            list[int]: The solution if found, empty list otherwise
            float: The objective value if found, INF otherwise
    """
    blp = MPS_to_BLP(Path(path))
    return solve_blp_with_ortools(blp)


def solve_blp_with_ortools(blp: BLP) -> Tuple[bool, list[int], float]:
    """Solve a BLP instance using OR-Tools.
    
    Args:
        blp: Binary Linear Programming instance
        
    Returns:
        tuple containing:
            bool: True if feasible solution found
            list[int]: The solution if found, empty list otherwise
            float: The objective value if found, INF otherwise
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        raise RuntimeError("Could not create SCIP solver")

    x = []
    for j in range(blp.n):
        x.append(solver.IntVar(0, 1, f'x[{j}]'))

    for i in range(blp.m):
        constraint = solver.RowConstraint(blp.b[i], blp.b[i], f'cover[{i}]')
        for j in range(blp.n):
            if blp.A[i, j] != 0:
                constraint.SetCoefficient(x[j], blp.A[i, j])

    objective = solver.Objective()
    for j in range(blp.n):
        if blp.c[j] != 0:
            objective.SetCoefficient(x[j], blp.c[j])
    objective.SetMinimization()

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        solution = [int(var.solution_value()) for var in x]
        value = float(solver.Objective().Value())
        return True, solution, value
    else:
        return False, [], float('inf') 