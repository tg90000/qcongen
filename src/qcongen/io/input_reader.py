from pathlib import Path
import logging
import numpy as np

from qcongen.opt_objects.bin_lp import BLP

logger = logging.getLogger('qcongen')

def MPS_to_BLP(path: Path) -> BLP:
    """Convert an MPS file to a BLP object.

    Parses an MPS format file containing a binary linear program:
    min c^T x
    s.t. Ax = b
         x binary

    Args:
        path: Path to the MPS file

    Returns:
        BLP: Binary Linear Program object with parsed data

    Raises:
        ValueError: If file format is invalid or constraints aren't equalities
        OSError: If file cannot be read
    """
    rows: dict[str, str] = {}
    coeffs: list[tuple[str, str, float]] = []
    rhs: dict[str, float] = {}
    col_names: list[str] = []
    row_names: list[str] = []
    obj_name: str = ""

    current_section = ""

    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("NAME"):
                    current_section = "NAME"
                    continue
                elif any(
                    line.startswith(s) for s in ["ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA"]
                ) and len(line.split()) == 1:
                    current_section = line.split()[0]
                    continue


                if current_section == "ROWS":
                    type_, name = line.split()
                    rows[name] = type_
                    if type_ == "N":
                        obj_name = name
                    else:
                        row_names.append(name)

                elif current_section == "COLUMNS":
                    fields = line.split()
                    col_name = fields[0]
                    if col_name not in col_names:
                        col_names.append(col_name)
                    
                    for i in range(1, len(fields), 2):
                        if i + 1 < len(fields):
                            row_name = fields[i]
                            value = float(fields[i + 1])
                            coeffs.append((col_name, row_name, value))

                elif current_section == "RHS":
                    fields = line.split()
                    if len(fields) >= 3:
                        row_name = fields[1]
                        value = float(fields[2])
                        rhs[row_name] = value
                        if len(fields) >= 5:
                            row_name = fields[3]
                            value = float(fields[4])
                            rhs[row_name] = value

    except Exception as e:
        raise ValueError(f"Failed to parse MPS file: {str(e)}") from e

    if any(type_ != "E" for name, type_ in rows.items() if name != obj_name):
        raise ValueError("All constraints must be equality constraints (type 'E')")

    m = len(row_names)
    n = len(col_names)

    logger.debug(f"Problem dimensions: {m} constraints, {n} variables")
    logger.debug(f"Row names: {row_names}")
    logger.debug(f"RHS values: {rhs}")

    A = np.zeros((m, n))
    b = np.zeros(m)
    c = np.zeros(n)

    for col_name, row_name, value in coeffs:
        if row_name == obj_name:
            j = col_names.index(col_name)
            c[j] = value

    for col_name, row_name, value in coeffs:
        if row_name != obj_name:
            i = row_names.index(row_name)
            j = col_names.index(col_name)
            A[i, j] = value

    for row_name, value in rhs.items():
        if row_name in row_names:
            i = row_names.index(row_name)
            b[i] = value
            
    return BLP(A, b, c)


def input_file_to_BLP(path: Path) -> BLP:
    """Read a simple matrix format file to create a BLP object.

    Input file format:
    n m
    A_11 A_12 ... A_1n
    A_21 A_22 ... A_2n
    ...
    A_m1 A_m2 ... A_mn
    b_1 b_2 ... b_m
    c_1 c_2 ... c_n

    Args:
        path: Path to input file

    Returns:
        BLP: Binary Linear Program object

    Raises:
        ValueError: If file format is invalid or dimensions don't match
        OSError: If file cannot be read
    """
    try:
        with open(path, "r") as f:
            n, m = map(int, f.readline().strip().split())

            A = np.zeros((m, n))
            for i in range(m):
                row = list(map(float, f.readline().strip().split()))
                if len(row) != n:
                    raise ValueError(f"Row {i+1} has {len(row)} elements, expected {n}")
                A[i] = row

            b_line = list(map(float, f.readline().strip().split()))
            if len(b_line) != m:
                raise ValueError(f"Vector b has {len(b_line)} elements, expected {m}")
            b = np.array(b_line)

            c_line = list(map(float, f.readline().strip().split()))
            if len(c_line) != n:
                raise ValueError(f"Vector c has {len(c_line)} elements, expected {n}")
            c = np.array(c_line)

            return BLP(A, b, c)

    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid file format: {str(e)}") from e
    except Exception as e:
        raise OSError(f"Failed to read input file: {str(e)}") from e
