"""Generators for creating problem instances."""

import random
from pathlib import Path
import numpy as np
from qcongen.opt_objects.bin_lp import BLP

def partition_set(elements: set[int], min_size: int, max_size: int) -> list[set[int]]:
    """Create a random partition of a set into subsets.
    
    Args:
        elements: Set of elements to partition
        min_size: Minimum subset size
        max_size: Maximum subset size
        
    Returns:
        List of sets that form a partition of the input set
    """
    result: list[set[int]] = []
    remaining = set(elements)
    
    while remaining:
        if len(remaining) <= max_size:
            result.append(remaining)
            break
            
        max_current_size = min(max_size, len(remaining) - min_size)
        if max_current_size < min_size:
            result.append(remaining)
            break
            
        size = random.randint(min_size, max_current_size)
        subset = set(random.sample(list(remaining), size))
        result.append(subset)
        remaining -= subset
    
    return result

def generate_set_partition_instance(
    n_sets: int,
    n_elements: int,
    min_set_size: int = 2,
    max_set_size: int = 5,
    min_cost: int = 1,
    max_cost: int = 100,
    instance_name: str = "generated_instance",
    output_dir: Path | None = None,
) -> BLP:
    """Generate a set partitioning problem instance.
    
    Args:
        n_sets: Number of sets (variables) required
        n_elements: Number of elements in the universe (determines number of constraints)
        min_set_size: Minimum number of elements in each set
        max_set_size: Maximum number of elements in each set
        min_cost: Minimum cost for a set
        max_cost: Maximum cost for a set
        instance_name: Name of the instance (will be used in file name)
        output_dir: Optional directory to save MPS file
        
    Returns:
        BLP: The generated binary linear programming instance
        
    The generated problem will ensure:
    1. At least one feasible solution exists (by construction)
    2. Each set has a size between min_set_size and max_set_size
    3. All elements appear in at least one set
    4. Costs are randomly assigned between min_cost and max_cost
    """
    if n_sets < 1 or n_elements < 1:
        raise ValueError("Invalid input parameters")
    if min_set_size > max_set_size:
        raise ValueError("min_set_size must be <= max_set_size")
    if min_cost > max_cost:
        raise ValueError("min_cost must be <= max_cost")
    if max_set_size > n_elements:
        max_set_size = n_elements

    universe = set(range(n_elements))

    all_sets: list[set[int]] = []
    feasible_partitions: list[list[set[int]]] = []
    
    max_attempts = 10
    attempt = 0
    while len(all_sets) < n_sets and attempt < max_attempts:
        partition = partition_set(universe, min_set_size, max_set_size)
        if len(all_sets) + len(partition) <= n_sets:
            feasible_partitions.append(partition)
            for s in partition:
                if s not in all_sets:
                    all_sets.append(s)
        attempt += 1

    max_attempts = 100
    attempt = 0
    while len(all_sets) < n_sets and attempt < max_attempts:
        size = random.randint(min_set_size, max_set_size)
        new_set = set(random.sample(list(universe), size))
        if new_set not in all_sets:
            all_sets.append(new_set)
        attempt += 1
        
    if len(all_sets) < n_sets:
        raise ValueError(f"Could not generate {n_sets} unique sets with the given parameters. Try increasing max_set_size or decreasing min_set_size.")

    costs = np.zeros(n_sets)
    for i in range(n_sets):
        costs[i] = random.randint(min_cost, max_cost)

    shuffled_indices = list(range(len(all_sets)))
    random.shuffle(shuffled_indices)
    all_sets = [all_sets[i] for i in shuffled_indices]
    costs = costs[shuffled_indices]

    A = np.zeros((n_elements, n_sets))
    for j, s in enumerate(all_sets):
        for elem in s:
            A[elem, j] = 1  

    b = np.ones(n_elements)

    blp = BLP(A, b, costs)

    for partition in feasible_partitions:
        solution = np.zeros(n_sets)
        for s in partition:
            idx = all_sets.index(s)
            solution[idx] = 1
        residuals = np.abs(A @ solution - b)
        assert np.all(residuals < 1e-10), "Generated solution is not feasible!"

    if output_dir is not None:
        output_path = output_dir / f"{instance_name}.mps"
        blp.toMPS(str(output_path))

    return blp

def generate_random_instance(
    n_sets: int = 15,
    n_elements: int = 25,
    instance_name: str = "random",
    output_dir: Path | None = None,
) -> BLP:
    """Generate a random set partitioning instance with default parameters.
    
    This is a convenience wrapper around generate_set_partition_instance with
    reasonable default parameters.
    
    Args:
        n_sets: Number of sets (default: 15)
        n_elements: Number of elements (default: 25)
        instance_name: Name of the instance (default: "random")
        output_dir: Optional directory to save MPS file
        
    Returns:
        BLP: The generated binary linear programming instance
    """
    return generate_set_partition_instance(
        n_sets=n_sets,
        n_elements=n_elements,
        min_set_size=1,
        max_set_size=10,
        min_cost=1,
        max_cost=100,
        instance_name=instance_name,
        output_dir=output_dir,
    ) 