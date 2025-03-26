"""Output writer module for QConGen."""

from datetime import datetime
from pathlib import Path

from qcongen.opt_objects.bin_lp import BLP


def create_output_directory() -> Path:
    """Create a timestamped output directory.
    
    Returns:
        Path: Path to the created output directory
    """
    base_dir = Path("results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def write_output(blp: BLP, solution: list[int], value: float, path: Path) -> None:
    """Write the output to a file.

    Args:
        blp: The BLP object
        solution: The solution vector
        value: The objective value of the solution
        path: The path to the output file
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Solution: {solution}\n")
        f.write(f"Value: {value}\n")
        f.write("\nMPS Format:\n")
        f.write("NAME          BLP\n")

        f.write("ROWS\n")
        f.write(" N  OBJ\n")
        for i in range(blp.A.shape[0]):
            f.write(f" E  R{i}\n")

        f.write("COLUMNS\n")
        for j in range(blp.A.shape[1]):
            if blp.c[j] != 0:
                f.write(f"    X{j}      OBJ       {blp.c[j]:.6f}\n")

        for i in range(blp.A.shape[0]):
            if blp.A[i, j] != 0:
                f.write(f"    X{j}      R{i}       {blp.A[i,j]:.6f}\n")

        f.write("RHS\n")
        for i in range(blp.A.shape[0]):
            if blp.b[i] != 0:
                f.write(f"    RHS       R{i}       {blp.b[i]:.6f}\n")

        f.write("BOUNDS\n")
        for j in range(blp.A.shape[1]):
            f.write(f" BV BND       X{j}\n")

        f.write("ENDATA")
